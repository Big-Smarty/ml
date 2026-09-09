//! Supplied host: adapted from projects/ch29–32. Learners edit kernels, not resource plumbing.
use crate::model::{validate_data, Example, Model, TrainingResult, INITIAL};

#[derive(Clone, Copy)]
pub struct Kernels {
    pub vector: &'static str,
    pub matmul: &'static str,
    pub reduce: &'static str,
    pub training: &'static str,
    pub fused: &'static str,
}

use std::{
    error::Error,
    io,
    sync::mpsc,
    time::{Duration, Instant},
};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: usize = 64;
// Change only this constant when changing the ch29 parallel shader workgroup size.
pub const VECTOR_WORKGROUP_SIZE: usize = 64;

pub struct Host {
    matmul_pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
    kernels: Kernels,
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
pub struct Measurement {
    pub output: Vec<f32>,
    pub cpu_wall_time: Duration,
    pub device_time_nanoseconds: Option<f64>,
    pub used_f16: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    Separate,
    FusedF32,
    Mixed,
}

impl Host {
    pub fn new(kernels: Kernels) -> Result<Self, Box<dyn Error>> {
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
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let scale = pipeline(
            &device,
            "scale",
            include_str!("support/scale.wgsl"),
            "scale",
        );
        let relu = pipeline(&device, "relu", include_str!("support/relu.wgsl"), "relu");
        let fused = pipeline(&device, "fused f32", kernels.fused, "fused");
        let fused_f16 = shader_f16.then(|| {
            pipeline(
                &device,
                "fused f16",
                include_str!("support/fused_f16.wgsl"),
                "fused_f16",
            )
        });
        let limits = device.limits();
        let matmul_pipeline = pipeline(&device, "matmul", kernels.matmul, "matmul");
        let reduce_pipeline = pipeline(&device, "reduce", kernels.reduce, "reduce");
        if let Some(error) = pollster::block_on(errors.pop()) {
            return Err(format!("shader/pipeline validation: {error}").into());
        }
        Ok(Self {
            matmul_pipeline,
            reduce_pipeline,
            kernels,
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

    pub fn affine_relu_with_gpu(
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

pub fn f16_arithmetic_is_safe(input: &[f32]) -> bool {
    input.iter().all(|value| value.abs() <= 40_000.0)
}

pub fn affine_relu_scalar(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|value| (value * 1.5 + 0.25).max(0.0))
        .collect()
}

impl Host {
    /// Multiplies row-major A `[m,k]` by B `[k,n]` and returns row-major `[m,n]`.
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let (a_len, b_len, c_len) = checked_matmul_lengths(m, k, n)?;
        self.check_buffer_len(a_len)?;
        self.check_buffer_len(b_len)?;
        self.check_buffer_len(c_len)?;
        let groups_x = n.div_ceil(16);
        let groups_y = m.div_ceil(16);
        let dispatch_limit = self.limits.max_compute_workgroups_per_dimension as usize;
        if groups_x > dispatch_limit || groups_y > dispatch_limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "matrix dispatch exceeds this device's workgroup-count limit",
            )
            .into());
        }
        if a.len() != a_len || b.len() != b_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "shape requires A length {a_len} and B length {b_len}; got {} and {}",
                    a.len(),
                    b.len()
                ),
            )
            .into());
        }
        if a.iter().chain(b).any(|value| !value.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "matrix entries must be finite",
            )
            .into());
        }
        let dimensions = [u32::try_from(m)?, u32::try_from(k)?, u32::try_from(n)?, 0];
        let a_buffer = storage_init(&self.device, "A", bytemuck::cast_slice(a), false);
        let b_buffer = storage_init(&self.device, "B", bytemuck::cast_slice(b), false);
        let c_bytes = bytes_for(c_len)?;
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C"),
            size: c_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let shape_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matrix dimensions"),
                contents: bytemuck::cast_slice(&dimensions),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let layout = self.matmul_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul bindings"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: shape_buffer.as_entire_binding(),
                },
            ],
        });
        self.dispatch_and_read(
            &self.matmul_pipeline,
            &bind_group,
            groups_x,
            groups_y,
            &c_buffer,
            c_len,
        )
    }

    /// Reduces finite values through as many workgroup stages as needed.
    pub fn reduce_sum(&self, values: &[f32]) -> Result<f32, Box<dyn Error>> {
        validate_reduction_values(values)?;
        self.check_buffer_len(values.len())?;
        let mut current = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reduction input"),
                contents: bytemuck::cast_slice(values),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let mut current_len = values.len();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("staged reduction commands"),
            });
        while current_len > 1 {
            let groups = current_len.div_ceil(512);
            if groups > self.limits.max_compute_workgroups_per_dimension as usize {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "reduction dispatch exceeds this device's limit",
                )
                .into());
            }
            let next = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduction partials"),
                size: bytes_for(groups)?,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let dispatch_params = [u32::try_from(current_len)?, 0, 0, 0];
            let dispatch_params_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("reduction length"),
                        contents: bytemuck::cast_slice(&dispatch_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reduction bindings"),
                layout: &self.reduce_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: current.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: next.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dispatch_params_buffer.as_entire_binding(),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.reduce_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(u32::try_from(groups)?, 1, 1);
            }
            current = next;
            current_len = groups;
        }
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduction readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&current, 0, &readback, 0, 4);
        self.queue.submit([encoder.finish()]);
        read_f32(&self.device, &readback, 1).map(|result| result[0])
    }

    fn check_buffer_len(&self, len: usize) -> Result<(), Box<dyn Error>> {
        let bytes = bytes_for(len)?;
        if len > u32::MAX as usize
            || bytes > self.limits.max_buffer_size
            || bytes > self.limits.max_storage_buffer_binding_size
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "matrix buffer exceeds this device's address or binding limit",
            )
            .into());
        }
        Ok(())
    }

    fn dispatch_and_read(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        groups_x: usize,
        groups_y: usize,
        output: &wgpu::Buffer,
        output_len: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let bytes = bytes_for(output_len)?;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chapter 30 commands"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(u32::try_from(groups_x)?, u32::try_from(groups_y)?, 1);
        }
        encoder.copy_buffer_to_buffer(output, 0, &readback, 0, bytes);
        self.queue.submit([encoder.finish()]);
        read_f32(&self.device, &readback, output_len)
    }
}

fn validate_reduction_values(values: &[f32]) -> Result<(), Box<dyn Error>> {
    if values.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "reduction requires at least one value",
        )
        .into());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "values must be finite").into());
    }
    Ok(())
}

fn checked_matmul_lengths(
    m: usize,
    k: usize,
    n: usize,
) -> Result<(usize, usize, usize), Box<dyn Error>> {
    if m == 0 || k == 0 || n == 0 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "m, k, and n must be nonzero").into(),
        );
    }
    let overflow = || io::Error::new(io::ErrorKind::InvalidInput, "matrix shape overflows usize");
    Ok((
        m.checked_mul(k).ok_or_else(overflow)?,
        k.checked_mul(n).ok_or_else(overflow)?,
        m.checked_mul(n).ok_or_else(overflow)?,
    ))
}

impl Host {
    pub fn vector(
        &self,
        left: &[f32],
        right: &[f32],
        scale: f32,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        crate::validate_pair(left, right)?;
        if !scale.is_finite() {
            return Err("scale must be finite".into());
        }
        let element_count = u32::try_from(left.len())?;
        let byte_len = u64::try_from(left.len().checked_mul(4).ok_or("buffer size overflow")?)?;
        let device = &self.device;
        let queue = &self.queue;
        let limits = device.limits();
        let groups = u32::try_from(dispatch_count(left.len(), VECTOR_WORKGROUP_SIZE))?;
        if byte_len > limits.max_buffer_size
            || byte_len > limits.max_storage_buffer_binding_size
            || groups > limits.max_compute_workgroups_per_dimension
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "vector exceeds this device's buffer or dispatch limit",
            )
            .into());
        }
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector addition"),
            source: wgpu::ShaderSource::Wgsl(self.kernels.vector.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vector addition pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("add"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(errors.pop()) {
            return Err(format!("vector shader: {error}").into());
        }
        let storage = wgpu::BufferUsages::STORAGE;
        let left_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("left"),
            contents: bytemuck::cast_slice(left),
            usage: storage,
        });
        let right_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("right"),
            contents: bytemuck::cast_slice(right),
            usage: storage,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: byte_len,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let dispatch_params = [element_count, scale.to_bits(), 0, 0];
        let dispatch_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dispatch parameters"),
            contents: bytemuck::cast_slice(&dispatch_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector bindings"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: left_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: right_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dispatch_params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vector addition commands"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, byte_len);
        queue.submit([encoder.finish()]);
        read_f32(device, &readback, left.len())
    }
}

impl Host {
    pub fn train(
        &self,
        data: &[Example],
        steps: usize,
        learning_rate: f32,
    ) -> Result<TrainingResult, Box<dyn Error>> {
        validate_data(data)
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
        if steps == 0 || !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "steps and learning_rate must be positive",
            )
            .into());
        }
        let sample_count = u32::try_from(data.len())?;
        let step_count = u32::try_from(steps)?;
        let hidden_len = data.len().checked_mul(4).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "hidden buffer size overflow")
        })?;
        let history_len = steps.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "loss history size overflow")
        })?;
        let device = &self.device;
        let queue = &self.queue;
        let limits = device.limits();
        for element_count in [
            data.len() * 2,
            hidden_len,
            data.len(),
            INITIAL.parameters.len(),
            history_len,
        ] {
            let bytes = bytes_for(element_count)?;
            if bytes > limits.max_buffer_size || bytes > limits.max_storage_buffer_binding_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "training buffer exceeds this device's limit",
                )
                .into());
            }
        }
        if sample_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "training dispatch exceeds this device's limit",
            )
            .into());
        }
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident training shader"),
            source: wgpu::ShaderSource::Wgsl(self.kernels.training.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("training layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("training pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let forward = pipeline("forward pipeline", "forward");
        let backward = pipeline("backward pipeline", "backward");
        let update = pipeline("update pipeline", "update");
        if let Some(error) = pollster::block_on(errors.pop()) {
            return Err(format!("training shader: {error}").into());
        }
        let inputs: Vec<f32> = data
            .iter()
            .flat_map(|(features, _)| features)
            .copied()
            .collect();
        let targets: Vec<f32> = data.iter().map(|(_, target)| *target).collect();
        let inputs = storage_init(device, "inputs", bytemuck::cast_slice(&inputs), false);
        let targets = storage_init(device, "targets", bytemuck::cast_slice(&targets), false);
        let parameters = storage_init(
            device,
            "parameters",
            bytemuck::cast_slice(&INITIAL.parameters),
            true,
        );
        let hidden = storage_empty(device, "hidden activations", hidden_len, false)?;
        let logits = storage_empty(device, "logits", data.len(), false)?;
        let gradients = storage_empty(device, "gradients", INITIAL.parameters.len(), false)?;
        let losses = storage_empty(device, "loss history", history_len, true)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resident training commands"),
        });
        for step in 0..=step_count {
            let config = [sample_count, step, learning_rate.to_bits(), 0];
            let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("step config"),
                contents: bytemuck::cast_slice(&config),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("training bindings"),
                layout: &layout,
                entries: &[
                    binding(0, &inputs),
                    binding(1, &targets),
                    binding(2, &parameters),
                    binding(3, &hidden),
                    binding(4, &logits),
                    binding(5, &gradients),
                    binding(6, &losses),
                    binding(7, &config_buffer),
                ],
            });
            dispatch(&mut encoder, &forward, &group, sample_count.div_ceil(64));
            dispatch(&mut encoder, &backward, &group, 1);
            if step < step_count {
                dispatch(&mut encoder, &update, &group, 1);
            }
        }
        let parameters_readback =
            map_buffer(device, "parameters readback", INITIAL.parameters.len())?;
        let losses_readback = map_buffer(device, "loss readback", history_len)?;
        encoder.copy_buffer_to_buffer(
            &parameters,
            0,
            &parameters_readback,
            0,
            bytes_for(INITIAL.parameters.len())?,
        );
        encoder.copy_buffer_to_buffer(&losses, 0, &losses_readback, 0, bytes_for(history_len)?);
        queue.submit([encoder.finish()]);
        let parameters = read_f32(device, &parameters_readback, INITIAL.parameters.len())?
            .try_into()
            .map_err(|_| io::Error::other("GPU returned the wrong parameter count"))?;
        Ok(TrainingResult {
            model: Model { parameters },
            losses: read_f32(device, &losses_readback, history_len)?,
        })
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage_init(device: &wgpu::Device, label: &str, bytes: &[u8], copy_src: bool) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | if copy_src {
                wgpu::BufferUsages::COPY_SRC
            } else {
                wgpu::BufferUsages::empty()
            },
    })
}

fn storage_empty(
    device: &wgpu::Device,
    label: &str,
    len: usize,
    copy_src: bool,
) -> Result<wgpu::Buffer, Box<dyn Error>> {
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes_for(len)?,
        usage: wgpu::BufferUsages::STORAGE
            | if copy_src {
                wgpu::BufferUsages::COPY_SRC
            } else {
                wgpu::BufferUsages::empty()
            },
        mapped_at_creation: false,
    }))
}

fn map_buffer(
    device: &wgpu::Device,
    label: &str,
    len: usize,
) -> Result<wgpu::Buffer, Box<dyn Error>> {
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes_for(len)?,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    }))
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    x: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(x, 1, 1);
}
