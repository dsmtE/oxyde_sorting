use log;

use oxyde::{
    wgpu, wgpu_utils::{
        self, binding_builder, buffers::{self, StagingBufferWrapper}, uniform_buffer::UniformBufferWrapper, ShaderComposer
    }
};

use oxyde_sorting::GpuCountingSortModule;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C)]
struct InitUniforms {
    current_time_ms: u32,
    init_method: u32,
    init_value: u32,
}

struct BuffersAndPipeline {
    value_buffer: wgpu::Buffer,
    count_buffer: wgpu::Buffer,
    counting_sort_module: GpuCountingSortModule,
    value_staging_buffer: StagingBufferWrapper<u32, true>,
    count_staging_buffer: StagingBufferWrapper<u32, true>,
    sorting_staging_buffer: StagingBufferWrapper<u32, true>,
    init_uniforms_buffer: UniformBufferWrapper<InitUniforms>,
    value_bind_group: wgpu::BindGroup,
    init_values_pipeline: wgpu::ComputePipeline,
}

fn init_buffers_and_pipeline(device: &wgpu::Device, size: u32, workgroup_size: u32) -> BuffersAndPipeline {
    let size_of_u32 = std::mem::size_of::<u32>() as u64;
    let value_buffer = buffers::create_buffer_for_size(
        &device,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        Some("values buffer"),
        size as u64 * size_of_u32,
    );
    let count_buffer = buffers::create_buffer_for_size(
        &device,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        Some("count buffer"),
        size as u64 * size_of_u32,
    );

    let counting_sort_module = GpuCountingSortModule::new(&device, &value_buffer, &count_buffer, workgroup_size).unwrap();

    let value_staging_buffer: StagingBufferWrapper<u32, true> = buffers::StagingBufferWrapper::new(&device, size as _);
    let count_staging_buffer: StagingBufferWrapper<u32, true> = buffers::StagingBufferWrapper::new(&device, size as _);
    let sorting_staging_buffer: StagingBufferWrapper<u32, true> = buffers::StagingBufferWrapper::new(&device, size as _);

    let init_uniforms_buffer = UniformBufferWrapper::new(
        &device,
        InitUniforms {
            current_time_ms: 0,
            init_method: 2,
            init_value: 0,
        },
        wgpu::ShaderStages::COMPUTE,
    );

    let value_bind_group_layout_with_desc = binding_builder::BindGroupLayoutBuilder::new()
        .add_binding_compute(wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        })
        .create(&device, None);

    let value_bind_group = binding_builder::BindGroupBuilder::new(&value_bind_group_layout_with_desc)
        .resource(value_buffer.as_entire_binding())
        .create(&device, Some("init random value bind_group"));

    let init_shader_module = &device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("init shader"),
        source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(
            ShaderComposer::new(include_str!("shaders/init.wgsl").into(), Some("init"))
                .with_shader_define("WORKGROUP_SIZE", workgroup_size.into())
                .build()
                .unwrap(),
        )),
    });

    let init_values_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("init values pipeline"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("init values pipeline layout"),
            bind_group_layouts: &[&value_bind_group_layout_with_desc.layout, init_uniforms_buffer.layout()],
            push_constant_ranges: &[],
        })),
        module: init_shader_module,
        entry_point: "main",
    });

    BuffersAndPipeline {
        value_buffer,
        count_buffer,
        counting_sort_module,
        value_staging_buffer,
        count_staging_buffer,
        sorting_staging_buffer,
        init_uniforms_buffer,
        value_bind_group,
        init_values_pipeline,
    }
}

fn init_render_instance_and_device() -> (wgpu_utils::render_handles::RenderInstance, usize) {
    let mut render_instance = wgpu_utils::render_handles::RenderInstance::new(None, None);

    // Force HighPerformance adapter for choosing Dedicated GPU as it seems to not work (Device lost: Dropped - Device is dying.) on integrated intel GPU (Intel(R) Iris(R) Xe Graphics)
    let device_handle_id = pollster::block_on(render_instance.device(None, Some(wgpu::PowerPreference::HighPerformance)))
        .expect("Device creation failed");

    let device = &render_instance.devices[device_handle_id].device;

    device.on_uncaptured_error(Box::new(|err| panic!("{}", err)));

    device.set_device_lost_callback(Box::new(|device_lost_reason, str| {
        match device_lost_reason {
            wgpu::DeviceLostReason::ReplacedCallback => {
                log::debug!("Device replace lost callback: {}", str);
            }
            _ => {
                panic!("Device lost: {:?} - {}", device_lost_reason, str);
            }
        }
    }));

    (render_instance, device_handle_id)
}

fn count_values(values: &[u32], count_size: usize) -> Vec<u32> {
    let mut count = vec![0u32; count_size];
    for value in values.iter() {
        count[*value as usize] += 1;
    }
    count
}

fn cpu_prefix_sum(count: &mut [u32]) {
    for i in 1..count.len() {
        count[i] += count[i - 1];
    }
}

fn sorting_id_sort_from_count(values: &[u32], count: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut sorting_id = vec![0u32; values.len()];
    let mut count_after_sort = count.to_vec();
    for (i, value) in values.iter().enumerate() {
        let value = *value as usize;
        sorting_id[count_after_sort[value] as usize - 1] = i as u32;
        count_after_sort[value] -= 1;
    }
    (sorting_id, count_after_sort)
}

fn counting_sort_on_cpu(values_slice: &[u32], count_size: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut count_cpu = count_values(values_slice, count_size);
    cpu_prefix_sum(count_cpu.as_mut_slice());
    let (sorting_id_cpu, count_after_sort_cpu) = sorting_id_sort_from_count(values_slice, &count_cpu);

    (count_cpu, sorting_id_cpu, count_after_sort_cpu)
}

fn is_sorted_by_id(values: &[u32], sorting_id: &[u32]) -> bool {
    for i in 1..sorting_id.len() {
        if values[sorting_id[i] as usize] < values[sorting_id[i - 1] as usize] {
            return false;
        }
    }
    true
}

#[test]
#[should_panic(expected = "SizeError(4096, 32)")]
fn wrong_workgroup_size() {
    let (render_instance, device_handle_id) = init_render_instance_and_device();

    let device = &render_instance.devices[device_handle_id].device;

    init_buffers_and_pipeline(device, 4096u32, 32u32);
}

#[test]
fn check_sorting() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .with_module_level("naga", log::LevelFilter::Info)
        .with_module_level("wgpu_core", log::LevelFilter::Info)
        .init().unwrap();

    let (render_instance, device_handle_id) = init_render_instance_and_device();
    let device_handle = &render_instance.devices[device_handle_id];

    let wgpu_utils::render_handles::DeviceHandle { device, queue, .. } = device_handle;

    let size = 8192u32;
    let workgroup_size = 128u32;

    let BuffersAndPipeline {
        value_buffer,
        count_buffer,
        counting_sort_module,
        mut value_staging_buffer,
        mut count_staging_buffer,
        mut sorting_staging_buffer,
        mut init_uniforms_buffer,
        value_bind_group,
        init_values_pipeline,
    } = init_buffers_and_pipeline(&device, size, workgroup_size);

    init_uniforms_buffer.content_mut().current_time_ms = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u32;
    init_uniforms_buffer.content_mut().init_method = 2u32;
    init_uniforms_buffer.update_content(&queue);

    let mut commands: Vec<wgpu::CommandBuffer> = vec![];

    let workgroup_size_x = (size as u32 + workgroup_size) / workgroup_size;

    {
        let mut init_values_command_encoder: wgpu::CommandEncoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("init values encoder") });

        {
            let init_pass = &mut init_values_command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Init values Pass"),
                timestamp_writes: None,
            });

            init_pass.set_pipeline(&init_values_pipeline);
            init_pass.set_bind_group(0, &value_bind_group, &[]);
            init_pass.set_bind_group(1, &init_uniforms_buffer.bind_group(), &[]);
            init_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
        }

        commands.push(init_values_command_encoder.finish());
    }

    {
        let mut counting_scan_command_encoder: wgpu::CommandEncoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Conting and scan encoder") });

        counting_sort_module.dispatch_work(&mut counting_scan_command_encoder, &count_buffer);
        commands.push(counting_scan_command_encoder.finish());
    }

    {
        let mut copy_buffer_command_encoder: wgpu::CommandEncoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Copy buffer encoder") });

        value_staging_buffer.encode_read(&mut copy_buffer_command_encoder, &value_buffer);
        count_staging_buffer.encode_read(&mut copy_buffer_command_encoder, &count_buffer);
        sorting_staging_buffer.encode_read(&mut copy_buffer_command_encoder, counting_sort_module.sorting_id_buffer());

        commands.push(copy_buffer_command_encoder.finish());
    }

    // See https://github.com/gfx-rs/wgpu/issues/3806
    let index = queue.submit(commands);
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));

    let (sender_value, receiver_value) = std::sync::mpsc::channel();
    value_staging_buffer.map_buffer(Some(move |result: Result<(), wgpu::BufferAsyncError>| {
        let _ = sender_value.send(result).unwrap();
    }));

    let (sender_count, receiver_count) = std::sync::mpsc::channel();
    count_staging_buffer.map_buffer(Some(move |result: Result<(), wgpu::BufferAsyncError>| {
        let _ = sender_count.send(result).unwrap();
    }));

    sorting_staging_buffer.map_buffer(None::<fn(Result<(), wgpu::BufferAsyncError>)>);

    // wait here for map_buffer to be finished (with wait the lock should be set successfully set)
    device.poll(wgpu::Maintain::Wait);

    // Read bufferAsyncError
    receiver_value
        .recv()
        .expect("MPSC channel must not fail")
        .expect("buffer reading value failed");
    receiver_count
        .recv()
        .expect("MPSC channel must not fail")
        .expect("buffer reading count failed");

    // Read buffer
    value_staging_buffer.read_and_unmap_buffer();
    count_staging_buffer.read_and_unmap_buffer();
    sorting_staging_buffer.read_and_unmap_buffer();

    let values_slice = value_staging_buffer.values_as_slice();

    // Do the same work as expected on CPU
    let (_, sorting_id_cpu, count_after_sort_cpu) = counting_sort_on_cpu(values_slice, size as usize);

    const MAX_TO_SHOW: usize = 64;
    println!("Size       : {} (show only first {} elements)", size, std::cmp::min(size as usize, MAX_TO_SHOW));
    println!("values     : {:?}", value_staging_buffer.iter().take(MAX_TO_SHOW).collect::<Vec<_>>());
    println!("GPU counts : {:?}", count_staging_buffer.iter().take(MAX_TO_SHOW).collect::<Vec<_>>());
    println!("CPU counts : {:?}", count_after_sort_cpu.iter().take(MAX_TO_SHOW).collect::<Vec<_>>());
    println!("GPU Sort   : {:?}", sorting_staging_buffer.iter().take(MAX_TO_SHOW).collect::<Vec<_>>());
    println!("CPU Sort   : {:?}", sorting_id_cpu.iter().take(MAX_TO_SHOW).collect::<Vec<_>>());

    let sorted_cpu = is_sorted_by_id(values_slice, &sorting_id_cpu);
    let sorted_gpu = is_sorted_by_id(values_slice, sorting_staging_buffer.values_as_slice());
    let count_after_sort_equal = count_after_sort_cpu == count_staging_buffer.values_as_slice();

    assert!(sorted_cpu, "CPU sorting is not correct");
    assert!(sorted_gpu, "GPU sorting is not correct");
    assert!(count_after_sort_equal, "CPU and GPU count after sort are not equal");

    // Clear device lost callback
    device.set_device_lost_callback(Box::new(|_, _| {}));
}
