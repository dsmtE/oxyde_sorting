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

pub struct App {
    size: u64,
    do_sorting: bool,

    value_buffer: wgpu::Buffer,

    staging_buffer: buffers::StagingBufferWrapper<u32, true>,
    init_random_value_bind_group: wgpu::BindGroup,

    init_random_pipeline: wgpu::ComputePipeline,
}

impl oxyde::App for App {
    fn create(_app_state: &mut AppState) -> Self {

        let device = &_app_state.device;

        let size = 1024u64;
        let buffer_size = size * std::mem::size_of::<u64>() as u64;
        
        let value_buffer = buffers::create_buffer_for_size(device, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, Some("value buffer"), buffer_size);

        let staging_buffer = buffers::StagingBufferWrapper::new(device, size as _);

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
                .add_shader_define("WORKGROUP_SIZE", 64.into())
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
        
        Self {
            size,
            do_sorting: false,
            value_buffer,
            staging_buffer,
            init_random_value_bind_group,
            init_random_pipeline,
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

                compute_pass.set_pipeline(&self.init_random_pipeline);
                compute_pass.set_bind_group(0, &self.init_random_value_bind_group, &[]);
                compute_pass.dispatch_workgroups(self.size as u32/64, 1, 1);
            }

            self.staging_buffer.encode_read(&mut compute_encoder, &self.value_buffer);
            
            // See https://github.com/gfx-rs/wgpu/issues/3806
            let index = app_state.queue.submit(Some(compute_encoder.finish()));
            app_state.device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));

            let (sender, receiver) = std::sync::mpsc::channel();
            {
                self.staging_buffer.map_buffer(Some(move |result: Result<(), wgpu::BufferAsyncError>| {
                    let _ = sender.send(result).unwrap();
                }));
            }

            // wait here for map_buffer to be finished (with wait the lock should be set successfully set)
            app_state.device.poll(wgpu::Maintain::Wait);
            
            // Read bufferAsyncError
            receiver.recv().expect("MPSC channel must not fail").expect("buffer reading failed");

            // Read buffer
            self.staging_buffer.read_and_unmap_buffer();

            println!("values: {:?}", self.staging_buffer.values_as_slice());

            self.do_sorting = false;
        }

        Ok(())
    }

    fn render(&mut self, _app_state: &mut AppState, _output_view: &wgpu::TextureView) -> Result<()> { Ok(()) }

    fn post_render(&mut self, _app_state: &mut AppState) -> Result<()> { Ok(()) }
}
