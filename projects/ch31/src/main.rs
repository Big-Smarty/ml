//! Device-resident forward, backward, and update passes for a tiny nonlinear network.

use std::{error::Error, io, sync::mpsc};
use wgpu::util::DeviceExt;

type Features = [f32; 2];
type Example = (Features, f32);

const TRAIN_DATA: [Example; 8] = [
    ([-1.0, -1.0], 0.0),
    ([-1.0, 1.0], 1.0),
    ([1.0, -1.0], 1.0),
    ([1.0, 1.0], 0.0),
    ([-0.8, -1.2], 0.0),
    ([-1.2, 0.8], 1.0),
    ([0.8, -1.2], 1.0),
    ([1.2, 0.8], 0.0),
];

#[derive(Clone, Debug)]
struct Model {
    // 0..8: hidden input weights; 8..12: hidden biases;
    // 12..16: output weights; 16: output bias.
    parameters: [f32; 17],
}

const INITIAL: Model = Model {
    parameters: [
        0.30, -0.20, -0.40, 0.35, 0.25, 0.45, -0.35, -0.25, 0.05, -0.05, 0.10, -0.10, 0.40, -0.30,
        0.25, -0.35, 0.0,
    ],
};

#[cfg(test)]
#[derive(Debug)]
struct Gradient {
    parameters: [f32; 17],
}

#[derive(Debug)]
struct TrainingResult {
    model: Model,
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

fn train_with_gpu(
    data: &[Example],
    steps: usize,
    learning_rate: f32,
) -> Result<TrainingResult, Box<dyn Error>> {
    validate_data(data).map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
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
    let history_len = steps
        .checked_add(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "loss history size overflow"))?;
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
    let inputs: Vec<f32> = data
        .iter()
        .flat_map(|(features, _)| features)
        .copied()
        .collect();
    let targets: Vec<f32> = data.iter().map(|(_, target)| *target).collect();
    let inputs = storage_init(&device, "inputs", bytemuck::cast_slice(&inputs), false);
    let targets = storage_init(&device, "targets", bytemuck::cast_slice(&targets), false);
    let parameters = storage_init(
        &device,
        "parameters",
        bytemuck::cast_slice(&INITIAL.parameters),
        true,
    );
    let hidden = storage_empty(&device, "hidden activations", hidden_len, false)?;
    let logits = storage_empty(&device, "logits", data.len(), false)?;
    let gradients = storage_empty(&device, "gradients", INITIAL.parameters.len(), false)?;
    let losses = storage_empty(&device, "loss history", history_len, true)?;
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
    let parameters_readback = map_buffer(&device, "parameters readback", INITIAL.parameters.len())?;
    let losses_readback = map_buffer(&device, "loss readback", history_len)?;
    encoder.copy_buffer_to_buffer(
        &parameters,
        0,
        &parameters_readback,
        0,
        bytes_for(INITIAL.parameters.len())?,
    );
    encoder.copy_buffer_to_buffer(&losses, 0, &losses_readback, 0, bytes_for(history_len)?);
    queue.submit([encoder.finish()]);
    let parameters = read_f32(&device, &parameters_readback, INITIAL.parameters.len())?
        .try_into()
        .map_err(|_| io::Error::other("GPU returned the wrong parameter count"))?;
    Ok(TrainingResult {
        model: Model { parameters },
        losses: read_f32(&device, &losses_readback, history_len)?,
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

impl Model {
    fn forward(&self, features: Features) -> ([f32; 4], f32) {
        let hidden = std::array::from_fn(|unit| {
            (self.parameters[unit * 2] * features[0]
                + self.parameters[unit * 2 + 1] * features[1]
                + self.parameters[8 + unit])
                .tanh()
        });
        let logit = self.parameters[16]
            + (0..4)
                .map(|unit| self.parameters[12 + unit] * hidden[unit])
                .sum::<f32>();
        (hidden, logit)
    }

    fn probability(&self, features: Features) -> f32 {
        sigmoid(self.forward(features).1)
    }

    #[cfg(test)]
    fn loss(&self, data: &[Example]) -> Result<f32, &'static str> {
        validate_data(data)?;
        let mut loss = 0.0;
        for &(features, target) in data {
            loss += binary_cross_entropy_from_logit(self.forward(features).1, target)?;
        }
        Ok(loss / data.len() as f32)
    }

    #[cfg(test)]
    fn gradient(&self, data: &[Example]) -> Result<Gradient, &'static str> {
        validate_data(data)?;
        let mut gradient = Gradient {
            parameters: [0.0; 17],
        };
        for &(features, target) in data {
            let (hidden, logit) = self.forward(features);
            let output_delta = sigmoid(logit) - target;
            for (unit, &activation) in hidden.iter().enumerate() {
                gradient.parameters[12 + unit] += output_delta * activation;
                let hidden_delta =
                    output_delta * self.parameters[12 + unit] * (1.0 - activation * activation);
                gradient.parameters[unit * 2] += hidden_delta * features[0];
                gradient.parameters[unit * 2 + 1] += hidden_delta * features[1];
                gradient.parameters[8 + unit] += hidden_delta;
            }
            gradient.parameters[16] += output_delta;
        }
        for value in &mut gradient.parameters {
            *value /= data.len() as f32;
        }
        Ok(gradient)
    }

    #[cfg(test)]
    fn step(&mut self, data: &[Example], learning_rate: f32) -> Result<(), &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning_rate must be finite and positive");
        }
        let gradient = self.gradient(data)?;
        for (parameter, slope) in self.parameters.iter_mut().zip(gradient.parameters) {
            *parameter -= learning_rate * slope;
        }
        Ok(())
    }

    #[cfg(test)]
    fn train(
        mut self,
        data: &[Example],
        steps: usize,
        learning_rate: f32,
    ) -> Result<Self, &'static str> {
        validate_data(data)?;
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning_rate must be finite and positive");
        }
        for _ in 0..steps {
            self.step(data, learning_rate)?;
        }
        Ok(self)
    }
}

fn sigmoid(logit: f32) -> f32 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exp = logit.exp();
        exp / (1.0 + exp)
    }
}

fn validate_data(data: &[Example]) -> Result<(), &'static str> {
    if data.is_empty() {
        return Err("training requires examples");
    }
    if data.iter().any(|(features, target)| {
        features.iter().any(|value| !value.is_finite())
            || !target.is_finite()
            || !(0.0..=1.0).contains(target)
    }) {
        return Err("features must be finite and targets must be in [0, 1]");
    }
    Ok(())
}

#[cfg(test)]
fn binary_cross_entropy_from_logit(logit: f32, target: f32) -> Result<f32, &'static str> {
    if !logit.is_finite() || !target.is_finite() || !(0.0..=1.0).contains(&target) {
        return Err("logit must be finite and target must be in [0, 1]");
    }
    Ok(logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p())
}

fn main() -> Result<(), Box<dyn Error>> {
    let steps = 800;
    let result = train_with_gpu(&TRAIN_DATA, steps, 0.3)?;
    println!("initial mean cross-entropy: {:.6}", result.losses[0]);
    println!("final mean cross-entropy:   {:.6}", result.losses[steps]);
    println!("parameters stayed on-device across {steps} forward/backward/update steps");
    for &(features, target) in &TRAIN_DATA[..4] {
        let probability = result.model.probability(features);
        println!(
            "point {:?} -> {:.4} (target {})",
            features, probability, target
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_gradient_matches_central_difference() -> Result<(), &'static str> {
        let analytic = INITIAL.gradient(&TRAIN_DATA)?;
        let epsilon = 1e-3;
        for index in [0, 7, 8, 12, 16] {
            let mut plus = INITIAL.clone();
            let mut minus = INITIAL.clone();
            plus.parameters[index] += epsilon;
            minus.parameters[index] -= epsilon;
            let numerical = (plus.loss(&TRAIN_DATA)? - minus.loss(&TRAIN_DATA)?) / (2.0 * epsilon);
            assert!(
                (analytic.parameters[index] - numerical).abs() < 2e-4,
                "index {index}: {} != {numerical}",
                analytic.parameters[index]
            );
        }
        Ok(())
    }

    #[test]
    fn scalar_training_fits_nonlinear_xor() -> Result<(), &'static str> {
        let trained = INITIAL.clone().train(&TRAIN_DATA, 800, 0.3)?;
        assert!(trained.loss(&TRAIN_DATA)? < INITIAL.loss(&TRAIN_DATA)? * 0.3);
        Ok(())
    }

    #[test]
    fn binary_cross_entropy_stays_finite_at_extremes() -> Result<(), &'static str> {
        assert!(binary_cross_entropy_from_logit(1000.0, 0.0)?.is_finite());
        assert!(binary_cross_entropy_from_logit(-1000.0, 1.0)?.is_finite());
        assert_eq!(binary_cross_entropy_from_logit(1000.0, 1.0)?, 0.0);
        assert_eq!(binary_cross_entropy_from_logit(-1000.0, 0.0)?, 0.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(sigmoid(1000.0), 1.0);
        assert_eq!(sigmoid(0.0), 0.5);
        Ok(())
    }

    #[test]
    fn gpu_training_rejects_invalid_inputs_before_device_lookup() {
        assert!(train_with_gpu(&[], 1, 0.3).is_err());
        assert!(train_with_gpu(&TRAIN_DATA, 0, 0.3).is_err());
        assert!(train_with_gpu(&TRAIN_DATA, 1, f32::NAN).is_err());
        assert!(INITIAL.clone().train(&[], 0, 0.3).is_err());
        assert!(INITIAL.clone().train(&TRAIN_DATA, 0, f32::NAN).is_err());
    }

    #[test]
    #[ignore = "requires a hardware Vulkan GPU"]
    fn gpu_training_matches_cpu_and_lowers_loss() -> Result<(), Box<dyn Error>> {
        let gpu = train_with_gpu(&TRAIN_DATA, 80, 0.3)?;
        let cpu = INITIAL.clone().train(&TRAIN_DATA, 80, 0.3)?;
        assert!(gpu.losses[80] < gpu.losses[0]);
        for (&actual, &expected) in gpu.model.parameters.iter().zip(&cpu.parameters) {
            assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
        }
        // A different row count proves both paths consume the supplied data.
        let data = &TRAIN_DATA[1..4];
        let gpu = train_with_gpu(data, 3, 0.3)?;
        let cpu = INITIAL.clone().train(data, 3, 0.3)?;
        for (&actual, &expected) in gpu.model.parameters.iter().zip(&cpu.parameters) {
            assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
        }
        assert!((gpu.losses[3] - cpu.loss(data)?).abs() < 2e-4);
        Ok(())
    }
}
