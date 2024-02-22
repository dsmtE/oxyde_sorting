use std::vec;

use oxyde::{
    anyhow::Result,
    egui,
    wgpu,
    wgpu_utils::{
        binding_builder,
        buffers,
        ShaderComposer,
    },
    winit::event::Event,
    AppState,
};


const WORKGROUP_SIZE: u32 = 64;

pub struct App {
    size: u32,
    do_sorting: bool,

    value_buffer: wgpu::Buffer,
    count_buffer: wgpu::Buffer,

    value_staging_buffer: buffers::StagingBufferWrapper<u32, true>,
    count_staging_buffer: buffers::StagingBufferWrapper<u32, true>,
    init_random_value_bind_group: wgpu::BindGroup,
    counting_bind_group: wgpu::BindGroup,

    init_random_pipeline: wgpu::ComputePipeline,
    counting_pipeline: wgpu::ComputePipeline,
    reset_pipeline: wgpu::ComputePipeline,
}

impl oxyde::App for App {
    fn create(_app_state: &mut AppState) -> Self {

        let device = &_app_state.device;

        let size = 16384u32;
        let buffer_size = size * std::mem::size_of::<u32>() as u32;
        
        let value_buffer = buffers::create_buffer_for_size(device, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, Some("value buffer"), buffer_size as _);
        let count_buffer = buffers::create_buffer_for_size(device, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, Some("sorting id buffer"), buffer_size as _);

        let value_staging_buffer = buffers::StagingBufferWrapper::new(device, size as _);
        let count_staging_buffer = buffers::StagingBufferWrapper::new(device, size as _);

        let init_random_value_bind_group_layout_with_desc = binding_builder::BindGroupLayoutBuilder::new()
                .add_binding_compute(wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                })
                .create(device, None);
        
        let init_random_value_bind_group = binding_builder::BindGroupBuilder::new(&init_random_value_bind_group_layout_with_desc)
                .resource(value_buffer.as_entire_binding())
                .create(device, Some("init random value bind_group"));

        let init_random_pipeline = {
            let source_naga_module = ShaderComposer::new(include_str!("../shaders/init_random.wgsl").into(), Some("init random"))
                .add_shader_define("WORKGROUP_SIZE", WORKGROUP_SIZE.into())
                .build()
                .unwrap();

            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("init random value pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("init random value pipeline layout"),
                    bind_group_layouts: &[&init_random_value_bind_group_layout_with_desc.layout],
                    push_constant_ranges: &[],
                })),
                module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("init random value shader"),
                    source : wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(source_naga_module)),
                }),
                entry_point: "main",
            })
        };


        let counting_bind_group_layout_with_desc = binding_builder::BindGroupLayoutBuilder::new()
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

        let counting_bind_group = binding_builder::BindGroupBuilder::new(&counting_bind_group_layout_with_desc)
            .resource(value_buffer.as_entire_binding())
            .resource(count_buffer.as_entire_binding())
            .create(device, Some("sorting bind_group"));
        

        let counting_naga_module = ShaderComposer::new(include_str!("../shaders/counting.wgsl").into(), Some("counting"))
            .add_shader_define("WORKGROUP_SIZE", WORKGROUP_SIZE.into())
            .build()
            .unwrap();
        
        let counting_shader_module = &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("counting shader"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(counting_naga_module)),
        });

        let counting_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("counting pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("counting pipeline layout"),
                bind_group_layouts: &[&counting_bind_group_layout_with_desc.layout],
                push_constant_ranges: &[],
            })),
            module: &counting_shader_module,
            entry_point: "count",
        });

        let reset_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("counting reset pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("counting reset pipeline layout"),
                bind_group_layouts: &[&counting_bind_group_layout_with_desc.layout],
                push_constant_ranges: &[],
            })),
            module: &counting_shader_module,
            entry_point: "reset",
        });
    
        
        Self {
            size,
            do_sorting: true,
            value_buffer,
            count_buffer,
            value_staging_buffer,
            count_staging_buffer,
            init_random_value_bind_group,
            counting_bind_group,
            init_random_pipeline,
            counting_pipeline,
            reset_pipeline,
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

            let mut compute_encoder: wgpu::CommandEncoder = app_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Sorting Encoder") });

            {
                let compute_pass = &mut compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compute Pass"), timestamp_writes: None });

                let workgroup_size_x = self.size as u32/WORKGROUP_SIZE;
                compute_pass.set_pipeline(&self.init_random_pipeline);
                compute_pass.set_bind_group(0, &self.init_random_value_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_size_x, 1, 1);

                compute_pass.set_pipeline(&self.reset_pipeline);
                compute_pass.set_bind_group(0, &self.counting_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_size_x, 1, 1);

                compute_pass.set_pipeline(&self.counting_pipeline);
                compute_pass.set_bind_group(0, &self.counting_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
            }

            self.value_staging_buffer.encode_read(&mut compute_encoder, &self.value_buffer);
            self.count_staging_buffer.encode_read(&mut compute_encoder, &self.count_buffer);
            
            // See https://github.com/gfx-rs/wgpu/issues/3806
            let index = app_state.queue.submit(Some(compute_encoder.finish()));
            app_state.device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));

            let (sender_value, receiver_value) = std::sync::mpsc::channel();
            self.value_staging_buffer.map_buffer(Some(move |result: Result<(), wgpu::BufferAsyncError>| {
                let _ = sender_value.send(result).unwrap();
            }));
            
            let (sender_count, receiver_count) = std::sync::mpsc::channel();
            self.count_staging_buffer.map_buffer(Some(move |result: Result<(), wgpu::BufferAsyncError>| {
                let _ = sender_count.send(result).unwrap();
            }));

            // wait here for map_buffer to be finished (with wait the lock should be set successfully set)
            app_state.device.poll(wgpu::Maintain::Wait);
            
            // Read bufferAsyncError
            receiver_value.recv().expect("MPSC channel must not fail").expect("buffer reading value failed");
            receiver_count.recv().expect("MPSC channel must not fail").expect("buffer reading count failed");

            // Read buffer
            self.value_staging_buffer.read_and_unmap_buffer();
            self.count_staging_buffer.read_and_unmap_buffer();
            
            if self.size <= 128 {
                println!("values: {:?}", self.value_staging_buffer.values_as_slice());
                println!("counts: {:?}", self.count_staging_buffer.values_as_slice());
            }

            {
                //check equality by doing same count on cpu
                let mut count_cpu = vec![0u32; self.size as usize];
                for value in self.value_staging_buffer.values_as_slice().iter() {
                    count_cpu[*value as usize] += 1;
                }

                println!("Count si valid: {}", count_cpu == self.count_staging_buffer.values_as_slice());
            }

            self.do_sorting = false;
        }

        Ok(())
    }

    fn render(&mut self, _app_state: &mut AppState, _output_view: &wgpu::TextureView) -> Result<()> { Ok(()) }

    fn post_render(&mut self, _app_state: &mut AppState) -> Result<()> { Ok(()) }
}
