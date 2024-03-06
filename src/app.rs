use std::vec;

use oxyde::{
    anyhow::Result,
    egui,
    wgpu::{self, CommandBuffer},
    wgpu_utils::{
        binding_builder, buffers, uniform_buffer::UniformBufferWrapper, ShaderComposer
    },
    winit::event::Event,
    AppState,
};

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C)]
struct InitUniforms {
    current_time_ms: u32,
    init_method: u32,
    init_value: u32,
}

const WORKGROUP_SIZE: u32 = 128;

pub struct App {
    size: u32,
    do_sorting: bool,

    init_uniforms_buffer: UniformBufferWrapper<InitUniforms>,

    value_buffer: wgpu::Buffer,
    count_buffer: wgpu::Buffer,
    sorting_buffer: wgpu::Buffer,

    value_staging_buffer: buffers::StagingBufferWrapper<u32, true>,
    count_staging_buffer: buffers::StagingBufferWrapper<u32, true>,
    sorting_staging_buffer : buffers::StagingBufferWrapper<u32, true>,

    value_bind_group: wgpu::BindGroup,
    counting_bind_group: wgpu::BindGroup,
    count_buffer_bind_group: wgpu::BindGroup,
    sorting_bind_group: wgpu::BindGroup,

    init_values_pipeline: wgpu::ComputePipeline,
    counting_pipeline: wgpu::ComputePipeline,
    workgroup_scan_pipeline: wgpu::ComputePipeline,
    workgroup_scan_level1_pipeline: wgpu::ComputePipeline,
    workgroup_propagate_pipeline: wgpu::ComputePipeline,
    sorting_pipeline: wgpu::ComputePipeline,
}

impl oxyde::App for App {
    fn create(_app_state: &mut AppState) -> Self {

        let device = &_app_state.device;

        let size = 8192u32;
        let size_of_u32 = std::mem::size_of::<u32>() as u64;
        
        let value_buffer = buffers::create_buffer_for_size(device, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, Some("values buffer"), size as u64 * size_of_u32);
        let count_buffer = buffers::create_buffer_for_size(device, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, Some("count buffer"), size as u64 * size_of_u32);
        
        let sorting_buffer= buffers::create_buffer_for_size(device, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC , Some("sorting buffer"), size as u64 * size_of_u32);
        let value_staging_buffer = buffers::StagingBufferWrapper::new(device, size as _);
        let count_staging_buffer = buffers::StagingBufferWrapper::new(device, size as _);
        let sorting_staging_buffer = buffers::StagingBufferWrapper::new(device, size as _);

        let init_uniforms_buffer = UniformBufferWrapper::new(
            device,
            InitUniforms{ current_time_ms: 0, init_method: 0, init_value: 0 },
            wgpu::ShaderStages::COMPUTE
        );

        // Bind groups
        let value_bind_group_layout_with_desc = binding_builder::BindGroupLayoutBuilder::new()
            .add_binding_compute(wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            })
            .create(device, None);
        
        let value_bind_group = binding_builder::BindGroupBuilder::new(&value_bind_group_layout_with_desc)
            .resource(value_buffer.as_entire_binding())
            .create(device, Some("init random value bind_group"));

        let read_write_bind_group_layout_with_desc = binding_builder::BindGroupLayoutBuilder::new()
        .add_binding_compute(wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                })
                .add_binding_compute(wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                })
                .create(device, None);
        
        let single_read_write_storage_buffer_bind_group_layout_with_desc = binding_builder::BindGroupLayoutBuilder::new()
        .add_binding_compute(wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            })
            .create(device, None);
        
        let counting_bind_group = binding_builder::BindGroupBuilder::new(&read_write_bind_group_layout_with_desc)
            .resource(value_buffer.as_entire_binding())
            .resource(count_buffer.as_entire_binding())
            .create(device, Some("counting_bind_group"));

        let sorting_bind_group = binding_builder::BindGroupBuilder::new(&single_read_write_storage_buffer_bind_group_layout_with_desc)
            .resource(sorting_buffer.as_entire_binding())
            .create(device, Some("sorting_bind_group"));
    
        let count_buffer_bind_group = binding_builder::BindGroupBuilder::new(&single_read_write_storage_buffer_bind_group_layout_with_desc)
            .resource(count_buffer.as_entire_binding())
            .create(device, Some("count_buffer_bind_group"));

        // Shaders modules
        let init_naga_module = ShaderComposer::new(include_str!("../shaders/init.wgsl").into(), Some("init"))
            .add_shader_define("WORKGROUP_SIZE", WORKGROUP_SIZE.into())
            .build()
            .unwrap();

        let init_shader_module = &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("init shader"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(init_naga_module)),
        });

        let counting_naga_module = ShaderComposer::new(include_str!("../shaders/counting.wgsl").into(), Some("counting"))
            .add_shader_define("WORKGROUP_SIZE", WORKGROUP_SIZE.into())
            .build()
            .unwrap();
        
        let counting_shader_module = &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("counting shader"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(counting_naga_module)),
        });

        let scan_source_str = include_str!("../shaders/scan.wgsl");

        let scan_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan shader"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(
                ShaderComposer::new(scan_source_str, Some("scan"))
                .add_shader_define("WORKGROUP_SIZE", WORKGROUP_SIZE.into())
                .add_shader_define("SCAN_LEVEL", 0.into())
                .build()
                .unwrap()
            )),
        });

        // Unable to use push_constant as it's not available in wgpu yet so we have to use a shader define for the scan level
        // Otherwise we could have used a uniform buffer to pass the scan level but this force use to submit the queue for each scan level
        let scan_level1_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan shader"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(
                ShaderComposer::new(scan_source_str, Some("scan"))
                .add_shader_define("WORKGROUP_SIZE", WORKGROUP_SIZE.into())
                .add_shader_define("SCAN_LEVEL", 1.into())
                .build()
                .unwrap()
            )),
        });

        let sorting_shader_module = &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("counting shader"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(
                ShaderComposer::new(include_str!("../shaders/sorting.wgsl").into(), Some("sorting"))
                    .add_shader_define("WORKGROUP_SIZE", WORKGROUP_SIZE.into())
                    .build()
                    .unwrap()
            )),
        });

        // Pipelines
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

        let counting_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("counting pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("counting pipeline layout"),
                bind_group_layouts: &[&read_write_bind_group_layout_with_desc.layout],
                push_constant_ranges: &[],
            })),
            module: &counting_shader_module,
            entry_point: "count",
        });
        
        let workgroup_scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("workgroup scan pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("workgroup scan pipeline layout"),
                bind_group_layouts: &[&single_read_write_storage_buffer_bind_group_layout_with_desc.layout],
                push_constant_ranges: &[],
            })),
            module: &scan_shader_module,
            entry_point: "workgroup_scan",
        });

        let workgroup_scan_level1_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("workgroup scan level1 pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("workgroup scan level1 pipeline layout"),
                bind_group_layouts: &[&single_read_write_storage_buffer_bind_group_layout_with_desc.layout],
                push_constant_ranges: &[],
            })),
            module: &scan_level1_shader_module,
            entry_point: "workgroup_scan",
        });

        let workgroup_propagate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("workgroup propagate pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("workgroup propagate pipeline layout"),
                bind_group_layouts: &[&single_read_write_storage_buffer_bind_group_layout_with_desc.layout],
                push_constant_ranges: &[],
            })),
            module: &scan_shader_module,
            entry_point: "workgroup_propagate",
        });

        let sorting_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sorting pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sorting pipeline layout"),
                bind_group_layouts: &[
                    &read_write_bind_group_layout_with_desc.layout,
                    &single_read_write_storage_buffer_bind_group_layout_with_desc.layout,
                ],
                push_constant_ranges: &[],
            })),
            module: &sorting_shader_module,
            entry_point: "sort",
        });

        Self {
            size,
            do_sorting: true,

            init_uniforms_buffer,

            value_buffer,
            count_buffer,
            sorting_buffer,

            value_staging_buffer,
            count_staging_buffer,
            sorting_staging_buffer,

            value_bind_group,
            counting_bind_group,
            count_buffer_bind_group,
            sorting_bind_group,
            
            init_values_pipeline,
            counting_pipeline,
            workgroup_scan_pipeline,
            workgroup_scan_level1_pipeline,
            workgroup_propagate_pipeline,
            sorting_pipeline,
        }
    }

    fn handle_event<T: 'static>(&mut self, _app_state: &mut AppState, _event: &Event<T>) -> Result<()> { Ok(()) }

    fn render_gui(&mut self, _app_state: &mut AppState) -> Result<()> {
        egui::SidePanel::right("right panel").resizable(true).show(_app_state.egui_renderer.context(), |ui| {
            if ui.button("do sorting").clicked() {
                self.do_sorting = true;
            }
        });

        Ok(())
    }

    fn update(&mut self, app_state: &mut AppState) -> Result<()> {

        if self.do_sorting {
            println!("do sorting");

            self.init_uniforms_buffer.content_mut().current_time_ms = std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_micros() as u32;
            self.init_uniforms_buffer.content_mut().init_method = 2u32;
            self.init_uniforms_buffer.update_content(&app_state.queue);

            let mut commands: Vec<CommandBuffer> = vec![];

            let workgroup_size_x = (self.size as u32 + WORKGROUP_SIZE)/ WORKGROUP_SIZE;

            {
                let mut init_values_command_encoder: wgpu::CommandEncoder = app_state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("init values encoder") });

                {
                    let init_pass = &mut init_values_command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Init values Pass"), timestamp_writes: None });

                    init_pass.set_pipeline(&self.init_values_pipeline);
                    init_pass.set_bind_group(0, &self.value_bind_group, &[]);
                    init_pass.set_bind_group(1, &self.init_uniforms_buffer.bind_group(), &[]);
                    init_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
                }
                
                commands.push(init_values_command_encoder.finish());
            }

            {
                let mut counting_scan_command_encoder: wgpu::CommandEncoder = app_state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Conting and scan encoder") });

                counting_scan_command_encoder.clear_buffer(&self.count_buffer, 0, None);

                {
                    let count_pass = &mut counting_scan_command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compting Pass"), timestamp_writes: None });

                    count_pass.set_pipeline(&self.counting_pipeline);
                    count_pass.set_bind_group(0, &self.counting_bind_group, &[]);
                    count_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
                }

                {
                    let scan_pass = &mut counting_scan_command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Scan Pass"), timestamp_writes: None });
                        
                    scan_pass.set_bind_group(0, &self.count_buffer_bind_group, &[]);

                    scan_pass.set_pipeline(&self.workgroup_scan_pipeline);
                    scan_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
                    scan_pass.set_pipeline(&self.workgroup_scan_level1_pipeline);
                    scan_pass.dispatch_workgroups(std::cmp::max(workgroup_size_x / WORKGROUP_SIZE, 1), 1, 1);
                    scan_pass.set_pipeline(&self.workgroup_propagate_pipeline);
                    scan_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
                }

                {
                    let sort_pass = &mut counting_scan_command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Sort Pass"), timestamp_writes: None });

                    sort_pass.set_pipeline(&self.sorting_pipeline);
                    sort_pass.set_bind_group(0, &self.counting_bind_group, &[]);
                    sort_pass.set_bind_group(1, &self.sorting_bind_group, &[]);
                    sort_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
                }

                commands.push(counting_scan_command_encoder.finish());
            }

            {
                let mut copy_buffer_command_encoder: wgpu::CommandEncoder = app_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Copy buffer encoder") });

                self.value_staging_buffer.encode_read(&mut copy_buffer_command_encoder, &self.value_buffer);
                self.count_staging_buffer.encode_read(&mut copy_buffer_command_encoder, &self.count_buffer);
                self.sorting_staging_buffer.encode_read(&mut copy_buffer_command_encoder, &self.sorting_buffer);

                commands.push(copy_buffer_command_encoder.finish());
            }
            
            // See https://github.com/gfx-rs/wgpu/issues/3806
            let index = app_state.queue.submit(commands);
            app_state.device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));

            let (sender_value, receiver_value) = std::sync::mpsc::channel();
            self.value_staging_buffer.map_buffer(Some(move |result: Result<(), wgpu::BufferAsyncError>| {
                let _ = sender_value.send(result).unwrap();
            }));
            
            let (sender_count, receiver_count) = std::sync::mpsc::channel();
            self.count_staging_buffer.map_buffer(Some(move |result: Result<(), wgpu::BufferAsyncError>| {
                let _ = sender_count.send(result).unwrap();
            }));

            self.sorting_staging_buffer.map_buffer(None::<fn(Result<(), wgpu::BufferAsyncError>)>);

            // wait here for map_buffer to be finished (with wait the lock should be set successfully set)
            app_state.device.poll(wgpu::Maintain::Wait);
            
            // Read bufferAsyncError
            receiver_value.recv().expect("MPSC channel must not fail").expect("buffer reading value failed");
            receiver_count.recv().expect("MPSC channel must not fail").expect("buffer reading count failed");

            // Read buffer
            self.value_staging_buffer.read_and_unmap_buffer();
            self.count_staging_buffer.read_and_unmap_buffer();
            self.sorting_staging_buffer.read_and_unmap_buffer();

            const MAX_TO_SHOW: usize = 128;
            
            println!("Size   : {} (show only first {} elements)", self.size, std::cmp::min(self.size as usize, MAX_TO_SHOW));
            println!("values : {:?}", self.value_staging_buffer.iter().take(MAX_TO_SHOW).collect::<Vec<_>>());
            println!("counts : {:?}", self.count_staging_buffer.iter().take(MAX_TO_SHOW).collect::<Vec<_>>());
            println!("Sort   : {:?}", self.sorting_staging_buffer.iter().take(MAX_TO_SHOW).collect::<Vec<_>>());

            {
                let values_slice = self.value_staging_buffer.values_as_slice();
                //check equality by doing same count on cpu
                let mut count_cpu = vec![0u32; self.size as usize];
                for value in values_slice.iter() {
                    count_cpu[*value as usize] += 1;
                }
                for i in 1..(self.size as usize) {
                    count_cpu[i] += count_cpu[i - 1];
                }
                
                let mut sorting_id_cpu = vec![0u32; self.size as usize];
                let mut count_after_sort_cpu = count_cpu.clone();
                for (i, value) in values_slice.iter().enumerate() {
                    let value = *value as usize;
                    sorting_id_cpu[count_after_sort_cpu[value] as usize - 1] = i as u32;
                    count_after_sort_cpu[value] -= 1;
                }
                
                //check if sorting is right
                let mut sorted_cpu = true;
                for i in 1..(self.size as usize) {
                    sorted_cpu &= values_slice[sorting_id_cpu[i] as usize] >= values_slice[sorting_id_cpu[i - 1] as usize];
                }

                let mut sorted_gpu = true;
                let sorting_id_slice = self.sorting_staging_buffer.values_as_slice();
                for i in 1..(self.size as usize) {
                    sorted_gpu &= values_slice[sorting_id_slice[i] as usize] >= values_slice[sorting_id_slice[i - 1] as usize];
                }

                // println!("counts scan : {:?}", count_cpu.as_slice());
                // println!("scan gpu    : {:?}", self.count_staging_buffer.values_as_slice());
                
                println!("count_cpu == gpu : {}", count_cpu == self.count_staging_buffer.values_as_slice());
                println!("count_after_sort_cpu == gpu : {}", count_after_sort_cpu == self.count_staging_buffer.values_as_slice());
                println!("sorted_cpu : {:?}", sorted_cpu);
                println!("sorted_gpu : {:?}", sorted_gpu);
            }

            self.do_sorting = false;
        }

        Ok(())
    }

    fn render(&mut self, _app_state: &mut AppState, _output_view: &wgpu::TextureView) -> Result<()> { Ok(()) }

    fn post_render(&mut self, _app_state: &mut AppState) -> Result<()> { Ok(()) }
}