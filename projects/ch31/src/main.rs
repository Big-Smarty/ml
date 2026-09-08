//! Device-resident forward, backward, and update passes for a tiny nonlinear network.

use std::{error::Error, io, sync::mpsc};
use wgpu::util::DeviceExt;

const INPUTS: [f32; 16] = [
    -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -0.8, -1.2, -1.2, 0.8, 0.8, -1.2, 1.2, 0.8,
];
const TARGETS: [f32; 8] = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
const INITIAL: [f32; 17] = [
    0.30, -0.20, -0.40, 0.35, 0.25, 0.45, -0.35, -0.25, 0.05, -0.05, 0.10, -0.10, 0.40, -0.30,
    0.25, -0.35, 0.0,
];

#[derive(Debug)]
struct TrainingResult {
    weights: Vec<f32>,
    losses: Vec<f32>,
}

fn select_adapter() -> Result<wgpu::Adapter, Box<dyn Error>> {
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
    adapters
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
            .into()
        })
}

fn gpu_train(epochs: usize, rate: f32) -> Result<TrainingResult, Box<dyn Error>> {
    if epochs == 0 || !rate.is_finite() || rate <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "epochs and rate must be positive",
        )
        .into());
    }
    let epoch_count = u32::try_from(epochs)?;
    let adapter = select_adapter()?;
    let info = adapter.get_info();
    eprintln!(
        "selected hardware adapter: {} ({:?})",
        info.name, info.backend
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("chapter 31 device"),
        ..Default::default()
    }))?;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resident training shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("train.wgsl").into()),
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
    let inputs = storage_init(&device, "inputs", bytemuck::cast_slice(&INPUTS), false);
    let targets = storage_init(&device, "targets", bytemuck::cast_slice(&TARGETS), false);
    let weights = storage_init(&device, "weights", bytemuck::cast_slice(&INITIAL), true);
    let hidden = storage_empty(&device, "hidden activations", INPUTS.len() / 2 * 4, false)?;
    let predictions = storage_empty(&device, "logits", TARGETS.len(), false)?;
    let gradients = storage_empty(&device, "gradients", INITIAL.len(), false)?;
    let losses = storage_empty(&device, "loss history", epochs + 1, true)?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("resident training commands"),
    });
    for epoch in 0..=epoch_count {
        let config = [u32::try_from(TARGETS.len())?, epoch, rate.to_bits(), 0];
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("epoch config"),
            contents: bytemuck::cast_slice(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("training bindings"),
            layout: &layout,
            entries: &[
                binding(0, &inputs),
                binding(1, &targets),
                binding(2, &weights),
                binding(3, &hidden),
                binding(4, &predictions),
                binding(5, &gradients),
                binding(6, &losses),
                binding(7, &config_buffer),
            ],
        });
        dispatch(
            &mut encoder,
            &forward,
            &group,
            u32::try_from(TARGETS.len())?.div_ceil(64),
        );
        dispatch(&mut encoder, &backward, &group, 1);
        if epoch < epoch_count {
            dispatch(&mut encoder, &update, &group, 1);
        }
    }
    let weights_readback = map_buffer(&device, "weights readback", INITIAL.len())?;
    let losses_readback = map_buffer(&device, "loss readback", epochs + 1)?;
    encoder.copy_buffer_to_buffer(&weights, 0, &weights_readback, 0, bytes_for(INITIAL.len())?);
    encoder.copy_buffer_to_buffer(&losses, 0, &losses_readback, 0, bytes_for(epochs + 1)?);
    queue.submit([encoder.finish()]);
    Ok(TrainingResult {
        weights: read_f32(&device, &weights_readback, INITIAL.len())?,
        losses: read_f32(&device, &losses_readback, epochs + 1)?,
    })
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

fn bytes_for(len: usize) -> Result<u64, Box<dyn Error>> {
    Ok(u64::try_from(len.checked_mul(4).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "buffer size overflow")
    })?)?)
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

fn read_f32(
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
    let values = bytemuck::cast_slice::<u8, f32>(&mapped)[..len].to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

fn cpu_forward(weights: &[f32], input: [f32; 2]) -> (Vec<f32>, f32) {
    let hidden: Vec<f32> = (0..4)
        .map(|unit| {
            (weights[unit * 2] * input[0] + weights[unit * 2 + 1] * input[1] + weights[8 + unit])
                .tanh()
        })
        .collect();
    let logit = weights[16]
        + (0..4)
            .map(|unit| weights[12 + unit] * hidden[unit])
            .sum::<f32>();
    (hidden, logit)
}

#[cfg(test)]
fn binary_cross_entropy_from_logit(logit: f32, target: f32) -> f32 {
    logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p()
}

#[cfg(test)]
fn cpu_loss(weights: &[f32]) -> f32 {
    TARGETS
        .iter()
        .enumerate()
        .map(|(sample, &target)| {
            let (_, logit) = cpu_forward(weights, [INPUTS[sample * 2], INPUTS[sample * 2 + 1]]);
            binary_cross_entropy_from_logit(logit, target)
        })
        .sum::<f32>()
        / TARGETS.len() as f32
}

#[cfg(test)]
fn cpu_gradient(weights: &[f32]) -> Vec<f32> {
    let mut gradient = vec![0.0; weights.len()];
    for (sample, &target) in TARGETS.iter().enumerate() {
        let input = [INPUTS[sample * 2], INPUTS[sample * 2 + 1]];
        let (hidden, logit) = cpu_forward(weights, input);
        let prediction = 1.0 / (1.0 + (-logit).exp());
        let output_delta = prediction - target;
        for unit in 0..4 {
            gradient[12 + unit] += output_delta * hidden[unit];
            let hidden_delta =
                output_delta * weights[12 + unit] * (1.0 - hidden[unit] * hidden[unit]);
            gradient[unit * 2] += hidden_delta * input[0];
            gradient[unit * 2 + 1] += hidden_delta * input[1];
            gradient[8 + unit] += hidden_delta;
        }
        gradient[16] += output_delta;
    }
    for value in &mut gradient {
        *value /= TARGETS.len() as f32;
    }
    gradient
}

#[cfg(test)]
fn cpu_train(epochs: usize, rate: f32) -> Vec<f32> {
    let mut weights = INITIAL.to_vec();
    for _ in 0..epochs {
        let gradient = cpu_gradient(&weights);
        for (weight, slope) in weights.iter_mut().zip(gradient) {
            *weight -= rate * slope;
        }
    }
    weights
}

fn main() -> Result<(), Box<dyn Error>> {
    let epochs = 800;
    let result = gpu_train(epochs, 0.3)?;
    println!("initial mean cross-entropy: {:.6}", result.losses[0]);
    println!("final mean cross-entropy:   {:.6}", result.losses[epochs]);
    println!("weights stayed on-device across {epochs} forward/backward/update steps");
    for sample in 0..4 {
        let (_, logit) = cpu_forward(
            &result.weights,
            [INPUTS[sample * 2], INPUTS[sample * 2 + 1]],
        );
        let prediction = 1.0 / (1.0 + (-logit).exp());
        println!(
            "point {:?} -> {:.4} (target {})",
            &INPUTS[sample * 2..sample * 2 + 2],
            prediction,
            TARGETS[sample]
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_gradient_matches_central_difference() {
        let analytic = cpu_gradient(&INITIAL);
        let epsilon = 1e-3;
        for index in [0, 7, 8, 12, 16] {
            let mut plus = INITIAL;
            let mut minus = INITIAL;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical = (cpu_loss(&plus) - cpu_loss(&minus)) / (2.0 * epsilon);
            assert!(
                (analytic[index] - numerical).abs() < 2e-4,
                "index {index}: {} != {numerical}",
                analytic[index]
            );
        }
    }

    #[test]
    fn scalar_training_fits_nonlinear_xor() {
        let trained = cpu_train(800, 0.3);
        assert!(cpu_loss(&trained) < cpu_loss(&INITIAL) * 0.3);
    }

    #[test]
    fn logit_loss_stays_finite_at_extremes() {
        assert!(binary_cross_entropy_from_logit(1000.0, 0.0).is_finite());
        assert!(binary_cross_entropy_from_logit(-1000.0, 1.0).is_finite());
        assert_eq!(binary_cross_entropy_from_logit(1000.0, 1.0), 0.0);
        assert_eq!(binary_cross_entropy_from_logit(-1000.0, 0.0), 0.0);
    }

    #[test]
    #[ignore = "requires a hardware Vulkan GPU"]
    fn gpu_training_matches_cpu_and_lowers_loss() -> Result<(), Box<dyn Error>> {
        let gpu = gpu_train(80, 0.3)?;
        let cpu = cpu_train(80, 0.3);
        assert!(gpu.losses[80] < gpu.losses[0]);
        for (&actual, &expected) in gpu.weights.iter().zip(&cpu) {
            assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
        }
        Ok(())
    }
}
