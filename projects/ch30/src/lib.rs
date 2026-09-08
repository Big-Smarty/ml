//! Tiled row-major GPU matrix multiplication and a staged workgroup reduction.

use std::{error::Error, io, sync::mpsc};
use wgpu::util::DeviceExt;

/// A persistent hardware Vulkan device, queue, and the chapter's compute pipelines.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    matmul_pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
}

impl Gpu {
    /// Discovers a hardware Vulkan adapter, preferring a discrete GPU.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        for adapter in &adapters {
            let info = adapter.get_info();
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
        eprintln!(
            "selected hardware adapter: {} ({:?})",
            info.name, info.backend
        );
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("chapter 30 device"),
                ..Default::default()
            }))?;
        let matmul_pipeline = create_pipeline(
            &device,
            "tiled matmul",
            include_str!("matmul.wgsl"),
            "matmul",
        );
        let reduce_pipeline = create_pipeline(
            &device,
            "workgroup reduction",
            include_str!("reduce.wgsl"),
            "reduce",
        );
        let limits = device.limits();
        Ok(Self {
            device,
            queue,
            limits,
            matmul_pipeline,
            reduce_pipeline,
        })
    }

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
        let a_buffer = storage_init(&self.device, "A", bytemuck::cast_slice(a));
        let b_buffer = storage_init(&self.device, "B", bytemuck::cast_slice(b));
        let c_bytes = bytes_for_f32(c_len)?;
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
                size: bytes_for_f32(groups)?,
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
        readback_f32(&self.device, &readback, 1).map(|result| result[0])
    }

    fn check_buffer_len(&self, len: usize) -> Result<(), Box<dyn Error>> {
        let bytes = bytes_for_f32(len)?;
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
        let bytes = bytes_for_f32(output_len)?;
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
        readback_f32(&self.device, &readback, output_len)
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

fn create_pipeline(
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

fn storage_init(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
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

fn bytes_for_f32(len: usize) -> Result<u64, Box<dyn Error>> {
    Ok(u64::try_from(len.checked_mul(4).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "buffer size overflow")
    })?)?)
}

fn readback_f32(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
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
    let result = bytemuck::cast_slice::<u8, f32>(&mapped)[..len].to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(result)
}

/// Scalar oracle with the same row-major contract as [`Gpu::matmul`].
pub fn matmul_scalar(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let (a_len, b_len, c_len) = checked_matmul_lengths(m, k, n)?;
    if a.len() != a_len || b.len() != b_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "matrix lengths do not match dimensions",
        )
        .into());
    }
    if a.iter().chain(b).any(|value| !value.is_finite()) {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "matrix entries must be finite").into(),
        );
    }
    let mut c = vec![0.0; c_len];
    for row in 0..m {
        for col in 0..n {
            c[row * n + col] = (0..k)
                .map(|inner| a[row * k + inner] * b[inner * n + col])
                .sum();
        }
    }
    Ok(c)
}

/// Scalar oracle for [`Gpu::reduce_sum`].
pub fn reduce_sum_scalar(values: &[f32]) -> Result<f32, Box<dyn Error>> {
    validate_reduction_values(values)?;
    Ok(values.iter().sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (&actual, &expected) in left.iter().zip(right) {
            let tolerance = 1e-5 + 1e-4 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn scalar_oracle_handles_nonsquare_odd_shapes() -> Result<(), Box<dyn Error>> {
        let a: Vec<f32> = (0..51).map(|i| i as f32 / 10.0 - 2.0).collect();
        let b: Vec<f32> = (0..85).map(|i| (i % 9) as f32 * 0.2 - 0.7).collect();
        let c = matmul_scalar(&a, &b, 3, 17, 5)?;
        assert_eq!(
            matmul_scalar(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                2,
                3,
                2
            )?,
            [58.0, 64.0, 139.0, 154.0]
        );
        assert_eq!(c.len(), 15);
        assert!(c.iter().all(|value| value.is_finite()));
        assert_eq!(reduce_sum_scalar(&[1.0, 2.0, 3.0])?, 6.0);
        assert!(reduce_sum_scalar(&[]).is_err());
        assert!(reduce_sum_scalar(&[f32::NAN]).is_err());
        Ok(())
    }

    #[test]
    fn invalid_dimensions_are_rejected_without_a_gpu() {
        assert!(checked_matmul_lengths(0, 2, 3).is_err());
        assert!(checked_matmul_lengths(usize::MAX, 2, 1).is_err());
    }

    #[test]
    #[ignore = "requires a hardware Vulkan GPU"]
    fn gpu_matmul_and_reduction_match_scalar_oracles() -> Result<(), Box<dyn Error>> {
        let gpu = Gpu::new()?;
        let a: Vec<f32> = (0..51).map(|i| i as f32 / 10.0 - 2.0).collect();
        let b: Vec<f32> = (0..85).map(|i| (i % 9) as f32 * 0.2 - 0.7).collect();
        close(
            &gpu.matmul(&a, &b, 3, 17, 5)?,
            &matmul_scalar(&a, &b, 3, 17, 5)?,
        );
        let a_large: Vec<f32> = (0..323).map(|i| i as f32 / 10.0 - 2.0).collect();
        let b_large: Vec<f32> = (0..627).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
        close(
            &gpu.matmul(&a_large, &b_large, 17, 19, 33)?,
            &matmul_scalar(&a_large, &b_large, 17, 19, 33)?,
        );
        let values: Vec<f32> = (0..777).map(|i| i as f32 * 0.125 - 1.0).collect();
        let expected = reduce_sum_scalar(&values)?;
        assert!((gpu.reduce_sum(&values)? - expected).abs() < 1e-4);
        Ok(())
    }
}
