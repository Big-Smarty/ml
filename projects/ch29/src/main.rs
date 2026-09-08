//! A checked vector addition dispatched to a hardware Vulkan adapter.

use std::{error::Error, io, sync::mpsc};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: usize = 64;
const ABS_TOLERANCE: f32 = 1e-6;
const REL_TOLERANCE: f32 = 1e-6;

fn validate_add_inputs(left: &[f32], right: &[f32]) -> io::Result<()> {
    if left.is_empty() || left.len() != right.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vectors must have the same nonzero length",
        ));
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "inputs must be finite",
        ));
    }
    Ok(())
}

fn add_scalar(left: &[f32], right: &[f32]) -> io::Result<Vec<f32>> {
    validate_add_inputs(left, right)?;
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect())
}

fn dispatch_count(element_count: usize, workgroup_size: usize) -> usize {
    element_count.div_ceil(workgroup_size)
}

fn close_f32(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= ABS_TOLERANCE + REL_TOLERANCE * expected.abs()
}

fn select_hardware_vulkan() -> Result<wgpu::Adapter, Box<dyn Error>> {
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
    adapters
        .into_iter()
        .filter(|adapter| {
            let info = adapter.get_info();
            info.backend == wgpu::Backend::Vulkan && info.device_type != wgpu::DeviceType::Cpu
        })
        .max_by_key(|adapter| {
            u8::from(adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        })
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "no hardware Vulkan adapter found; software fallback is intentionally disabled",
            )
            .into()
        })
}

fn read_f32(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    element_count: usize,
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
    let values = bytemuck::cast_slice::<u8, f32>(&mapped)[..element_count].to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

fn add_with_gpu(left: &[f32], right: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    validate_add_inputs(left, right)?;
    let element_count = u32::try_from(left.len())?;
    let byte_len = u64::try_from(left.len().checked_mul(4).ok_or("buffer size overflow")?)?;
    let adapter = select_hardware_vulkan()?;
    let info = adapter.get_info();
    eprintln!(
        "selected hardware adapter: {} ({:?})",
        info.name, info.backend
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("chapter 29 device"),
        ..Default::default()
    }))?;
    let limits = device.limits();
    let groups = u32::try_from(dispatch_count(left.len(), WORKGROUP_SIZE))?;
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
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vector addition"),
        source: wgpu::ShaderSource::Wgsl(include_str!("add.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vector addition pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("add"),
        compilation_options: Default::default(),
        cache: None,
    });
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
    let dispatch_params = [element_count, 0, 0, 0];
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
    read_f32(&device, &readback, left.len())
}

fn main() -> Result<(), Box<dyn Error>> {
    let left = [1.0, -2.0, 3.5, 8.0, 0.25];
    let right = [2.0, 5.0, -0.5, 1.0, 0.75];
    let expected = add_scalar(&left, &right)?;
    let output = add_with_gpu(&left, &right)?;
    if output.len() != expected.len()
        || output
            .iter()
            .zip(&expected)
            .any(|(&actual, &expected)| !close_f32(actual, expected))
    {
        return Err(io::Error::other("GPU output disagrees with the scalar oracle").into());
    }
    println!("GPU result: {output:?}");
    println!("CPU scalar oracle: {expected:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_shapes_are_rejected_before_adapter_discovery() {
        assert!(add_with_gpu(&[], &[]).is_err());
        assert!(add_with_gpu(&[f32::NAN], &[1.0]).is_err());
        assert!(add_with_gpu(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn scalar_oracle_and_dispatch_cover_the_primitive() -> io::Result<()> {
        assert_eq!(add_scalar(&[1.0, -2.0], &[2.0, 5.0])?, [3.0, 3.0]);
        assert_eq!(dispatch_count(64, WORKGROUP_SIZE), 1);
        assert_eq!(dispatch_count(67, WORKGROUP_SIZE), 2);
        assert!(close_f32(10.000_01, 10.0));
        assert!(!close_f32(10.001, 10.0));
        Ok(())
    }

    #[test]
    #[ignore = "requires a hardware Vulkan GPU"]
    fn gpu_matches_scalar_oracle_for_partial_workgroup() -> Result<(), Box<dyn Error>> {
        let left: Vec<f32> = (0..67).map(|i| i as f32 * 0.25).collect();
        let right: Vec<f32> = (0..67).map(|i| 10.0 - i as f32 * 0.5).collect();
        let expected = add_scalar(&left, &right)?;
        let actual = add_with_gpu(&left, &right)?;
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert!(close_f32(actual, expected));
        }
        Ok(())
    }
}
