//! Measures separate and fused GPU kernels and selects an optional f16 arithmetic path.

use std::{
    error::Error,
    io,
    sync::mpsc,
    time::{Duration, Instant},
};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: usize = 64;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    scale: wgpu::ComputePipeline,
    relu: wgpu::ComputePipeline,
    fused: wgpu::ComputePipeline,
    fused_f16: Option<wgpu::ComputePipeline>,
    timestamp_queries: bool,
    limits: wgpu::Limits,
}

#[derive(Debug)]
struct Measurement {
    output: Vec<f32>,
    cpu_wall_time: Duration,
    device_time_nanoseconds: Option<f64>,
    used_f16: bool,
}

#[derive(Clone, Copy)]
enum Plan {
    Separate,
    FusedF32,
    Mixed,
}

impl Gpu {
    fn new() -> Result<Self, Box<dyn Error>> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        for candidate in &adapters {
            let info = candidate.get_info();
            eprintln!(
                "discovered adapter: {} ({:?}, {:?}) driver={} {}",
                info.name, info.backend, info.device_type, info.driver, info.driver_info
            );
        }
        let adapter = adapters
            .into_iter()
            .filter(|candidate| {
                let info = candidate.get_info();
                info.backend == wgpu::Backend::Vulkan && info.device_type != wgpu::DeviceType::Cpu
            })
            .max_by_key(|candidate| {
                u8::from(candidate.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::Unsupported,
                    "no hardware Vulkan adapter found; software fallback is intentionally disabled",
                )
            })?;
        let info = adapter.get_info();
        let available = adapter.features();
        let timestamp_queries = available.contains(wgpu::Features::TIMESTAMP_QUERY);
        let shader_f16 = available.contains(wgpu::Features::SHADER_F16);
        eprintln!(
            "selected hardware adapter: {} ({:?})",
            info.name, info.backend
        );
        eprintln!(
            "optional features: timestamp_query={timestamp_queries}, shader_f16={shader_f16}"
        );
        let mut required_features = wgpu::Features::empty();
        if timestamp_queries {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if shader_f16 {
            required_features |= wgpu::Features::SHADER_F16;
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("chapter 32 device"),
                required_features,
                ..Default::default()
            }))?;
        let scale = pipeline(&device, "scale", include_str!("scale.wgsl"), "scale");
        let relu = pipeline(&device, "relu", include_str!("relu.wgsl"), "relu");
        let fused = pipeline(&device, "fused f32", include_str!("fused.wgsl"), "fused");
        let fused_f16 = shader_f16.then(|| {
            pipeline(
                &device,
                "fused f16",
                include_str!("fused_f16.wgsl"),
                "fused_f16",
            )
        });
        let limits = device.limits();
        Ok(Self {
            device,
            queue,
            scale,
            relu,
            fused,
            fused_f16,
            timestamp_queries,
            limits,
        })
    }

    fn affine_relu_with_gpu(
        &self,
        input: &[f32],
        plan: Plan,
    ) -> Result<Measurement, Box<dyn Error>> {
        if input.is_empty() || input.iter().any(|value| !value.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input must be nonempty and finite",
            )
            .into());
        }
        let element_count = u32::try_from(input.len())?;
        let bytes = bytes_for(input.len())?;
        if bytes > self.limits.max_buffer_size
            || bytes > self.limits.max_storage_buffer_binding_size
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input exceeds this device's buffer limit",
            )
            .into());
        }
        let groups = u32::try_from(dispatch_count(input.len(), WORKGROUP_SIZE))?;
        if groups > self.limits.max_compute_workgroups_per_dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "dispatch exceeds this device's workgroup-count limit",
            )
            .into());
        }
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let intermediate = storage_output(&self.device, "intermediate", bytes, false);
        let output = storage_output(&self.device, "output", bytes, true);
        let dispatch_params = [element_count, 0, 0, 0];
        let dispatch_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("dispatch parameters"),
                    contents: bytemuck::cast_slice(&dispatch_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
        let readback = readback_buffer(&self.device, "output readback", bytes);
        let pass_count = if matches!(plan, Plan::Separate) { 2 } else { 1 };
        let query_count = pass_count * 2;
        let query = self.timestamp_queries.then(|| {
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: query_count,
            })
        });
        let query_resolve = query.as_ref().map(|_| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp resolve"),
                size: u64::from(query_count) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let query_readback = query.as_ref().map(|_| {
            readback_buffer(
                &self.device,
                "timestamp readback",
                u64::from(query_count) * 8,
            )
        });
        let use_f16 = matches!(plan, Plan::Mixed)
            && self.fused_f16.is_some()
            && f16_arithmetic_is_safe(input);
        let cpu_wall_started = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timed commands"),
            });
        match plan {
            Plan::Separate => {
                let first = group(
                    &self.device,
                    &self.scale,
                    &input_buffer,
                    &intermediate,
                    &dispatch_params_buffer,
                );
                let second = group(
                    &self.device,
                    &self.relu,
                    &intermediate,
                    &output,
                    &dispatch_params_buffer,
                );
                record_pass(
                    &mut encoder,
                    &self.scale,
                    &first,
                    groups,
                    query.as_ref().map(|set| (set, 0, 1)),
                );
                record_pass(
                    &mut encoder,
                    &self.relu,
                    &second,
                    groups,
                    query.as_ref().map(|set| (set, 2, 3)),
                );
            }
            Plan::FusedF32 => {
                let fused = group(
                    &self.device,
                    &self.fused,
                    &input_buffer,
                    &output,
                    &dispatch_params_buffer,
                );
                record_pass(
                    &mut encoder,
                    &self.fused,
                    &fused,
                    groups,
                    query.as_ref().map(|set| (set, 0, 1)),
                );
            }
            Plan::Mixed => {
                let selected = if use_f16 {
                    self.fused_f16.as_ref().unwrap_or(&self.fused)
                } else {
                    &self.fused
                };
                let fused = group(
                    &self.device,
                    selected,
                    &input_buffer,
                    &output,
                    &dispatch_params_buffer,
                );
                record_pass(
                    &mut encoder,
                    selected,
                    &fused,
                    groups,
                    query.as_ref().map(|set| (set, 0, 1)),
                );
            }
        }
        if let (Some(set), Some(resolve), Some(staging)) = (&query, &query_resolve, &query_readback)
        {
            encoder.resolve_query_set(set, 0..query_count, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, staging, 0, u64::from(query_count) * 8);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let output = read_f32(&self.device, &readback, input.len())?;
        let cpu_wall_time = cpu_wall_started.elapsed();
        let device_time_nanoseconds = if let Some(staging) = query_readback {
            let values = read_u64(&self.device, &staging, query_count as usize)?;
            let ticks = values
                .chunks_exact(2)
                .map(|pair| pair[1].wrapping_sub(pair[0]))
                .sum::<u64>();
            Some(ticks as f64 * f64::from(self.queue.get_timestamp_period()))
        } else {
            None
        };
        Ok(Measurement {
            output,
            cpu_wall_time,
            device_time_nanoseconds,
            used_f16: use_f16,
        })
    }
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    entry: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn storage_output(device: &wgpu::Device, label: &str, size: u64, copy_src: bool) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | if copy_src {
                wgpu::BufferUsages::COPY_SRC
            } else {
                wgpu::BufferUsages::empty()
            },
        mapped_at_creation: false,
    })
}

fn readback_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn group<'a>(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    input: &'a wgpu::Buffer,
    output: &'a wgpu::Buffer,
    dispatch_params: &'a wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kernel bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dispatch_params.as_entire_binding(),
            },
        ],
    })
}

fn record_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    groups: u32,
    timestamps: Option<(&wgpu::QuerySet, u32, u32)>,
) {
    let timestamp_writes =
        timestamps.map(|(query_set, start, end)| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(start),
            end_of_pass_write_index: Some(end),
        });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("measured pass"),
        timestamp_writes,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

fn bytes_for(len: usize) -> Result<u64, Box<dyn Error>> {
    Ok(u64::try_from(len.checked_mul(4).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "buffer size overflow")
    })?)?)
}

fn dispatch_count(element_count: usize, workgroup_size: usize) -> usize {
    element_count.div_ceil(workgroup_size)
}

fn map_bytes(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, Box<dyn Error>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _send_result = sender.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver
        .recv()
        .map_err(|_| io::Error::other("GPU mapping callback did not run"))??;
    let mapped = slice.get_mapped_range();
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(bytes)
}

fn read_f32(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = map_bytes(device, buffer)?;
    Ok(bytes
        .chunks_exact(4)
        .take(len)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read_u64(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let bytes = map_bytes(device, buffer)?;
    Ok(bytes
        .chunks_exact(8)
        .take(len)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

fn f16_arithmetic_is_safe(input: &[f32]) -> bool {
    input.iter().all(|value| value.abs() <= 40_000.0)
}

fn affine_relu_scalar(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|value| (value * 1.5 + 0.25).max(0.0))
        .collect()
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tolerance, "{a} != {e}");
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let gpu = Gpu::new()?;
    let input: Vec<f32> = (0..65_537).map(|i| (i % 257) as f32 / 32.0 - 4.0).collect();
    for plan in [Plan::Separate, Plan::FusedF32, Plan::Mixed] {
        for _ in 0..3 {
            let _warmup = gpu.affine_relu_with_gpu(&input, plan)?;
        }
    }
    let mut separate = Vec::new();
    let mut fused = Vec::new();
    let mut mixed = Vec::new();
    for _ in 0..7 {
        separate.push(gpu.affine_relu_with_gpu(&input, Plan::Separate)?);
        fused.push(gpu.affine_relu_with_gpu(&input, Plan::FusedF32)?);
        mixed.push(gpu.affine_relu_with_gpu(&input, Plan::Mixed)?);
    }
    let expected = affine_relu_scalar(&input);
    for sample in &separate {
        close(&sample.output, &expected, 1e-5);
    }
    for sample in &fused {
        close(&sample.output, &expected, 1e-5);
    }
    for sample in &mixed {
        close(
            &sample.output,
            &expected,
            if sample.used_f16 { 0.01 } else { 1e-5 },
        );
    }
    print_summary("separate f32 passes", &separate);
    print_summary("fused f32 pass", &fused);
    print_summary("mixed precision pass", &mixed);
    println!(
        "mixed path: {}",
        if mixed[0].used_f16 {
            "f16 arithmetic supported, in range, and selected"
        } else {
            "f32 fallback selected"
        }
    );
    Ok(())
}

fn print_summary(label: &str, measurements: &[Measurement]) {
    let mut walls: Vec<_> = measurements
        .iter()
        .map(|sample| sample.cpu_wall_time)
        .collect();
    walls.sort_unstable();
    let device_times: Option<Vec<f64>> = measurements
        .iter()
        .map(|sample| sample.device_time_nanoseconds)
        .collect();
    println!(
        "{label}: CPU wall-clock median={:?}, range={:?}..{:?}",
        walls[walls.len() / 2],
        walls[0],
        walls[walls.len() - 1]
    );
    if let Some(mut times) = device_times {
        times.sort_by(f64::total_cmp);
        println!(
            "{label}: summed device pass timestamps median={:.0} ns, range={:.0}..{:.0} ns",
            times[times.len() / 2],
            times[0],
            times[times.len() - 1]
        );
    } else {
        println!("{label}: device timestamps unavailable");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_relu_scalar_covers_relu_boundary() {
        assert_eq!(
            affine_relu_scalar(&[-1.0, -1.0 / 6.0, 0.0, 2.0]),
            vec![0.0, 0.0, 0.25, 3.25]
        );
        assert_eq!(dispatch_count(65_537, WORKGROUP_SIZE), 1_025);
    }

    #[test]
    fn mixed_precision_range_guard_rejects_half_overflow_risk() {
        assert!(f16_arithmetic_is_safe(&[-40_000.0, 0.0, 40_000.0]));
        assert!(!f16_arithmetic_is_safe(&[50_000.0]));
    }

    #[test]
    #[ignore = "requires a hardware Vulkan GPU"]
    fn separate_fused_and_mixed_paths_match() -> Result<(), Box<dyn Error>> {
        let gpu = Gpu::new()?;
        let input: Vec<f32> = (0..257).map(|i| i as f32 / 29.0 - 4.0).collect();
        let expected = affine_relu_scalar(&input);
        close(
            &gpu.affine_relu_with_gpu(&input, Plan::Separate)?.output,
            &expected,
            1e-5,
        );
        close(
            &gpu.affine_relu_with_gpu(&input, Plan::FusedF32)?.output,
            &expected,
            1e-5,
        );
        close(
            &gpu.affine_relu_with_gpu(&input, Plan::Mixed)?.output,
            &expected,
            if gpu.fused_f16.is_some() { 0.01 } else { 1e-5 },
        );
        Ok(())
    }
}
