use oxyde::{
    anyhow::Result,
    wgpu,
    wgpu_utils::{binding_builder, buffers, ShaderComposer},
    log,
};

// Structure that handle the counting and sorting of a buffer of u32
// The counting sorting is done in place and the sorting id are stored in a separate buffer
// The counting sort is done in 3 steps:
// 1. Counting the number of elements in each bucket
// 2. Scanning (prefix sum) the count buffer to get the starting index of each bucket
// 3. Write ids using the count buffer (atomic operation) to know where to write the id in the sorting id buffer
//
// The counting sort isn't stable as the last step is done in parallel and the order of the elements in the same bucket isn't preserved during this step
//
// The Scan part is done using the Kogge-Stone method at the workgroup level
// then using the strategy of "scan then propagate" by doing a second scan on the bigger values of each previous workgroup then propagating those values to get the final scan
pub struct GpuCountingSortModule {
    workgroup_size: u32,
    size: u32,

    sorting_id_buffer: wgpu::Buffer,

    counting_bind_group: wgpu::BindGroup,
    sorting_bind_group: wgpu::BindGroup,
    count_buffer_bind_group: wgpu::BindGroup,

    counting_pipeline: wgpu::ComputePipeline,
    workgroup_scan_pipeline: wgpu::ComputePipeline,
    workgroup_scan_level1_pipeline: wgpu::ComputePipeline,
    workgroup_propagate_pipeline: wgpu::ComputePipeline,
    sorting_pipeline: wgpu::ComputePipeline,
}

#[derive(Debug)]
pub enum CountingSortingError {
    MissingBufferUsage(wgpu::BufferUsages, &'static str),
    SizeError(u32, u32),
}

impl std::fmt::Display for CountingSortingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CountingSortingError::MissingBufferUsage(buffer_usage, buffer_name) =>
                write!(f, "Missing buffer usage {:?} for {}", buffer_usage, buffer_name),
            CountingSortingError::SizeError(size, workgroup_size) => {
                write!(
                    f,
                    "Unable to handle a buffer of size {} with a workgroup size of {} (Current limitation workgroup_size*workgroup_size : {})",
                    size,
                    workgroup_size,
                    workgroup_size * workgroup_size
                )
            },
        }
    }
}

impl std::error::Error for CountingSortingError {}

impl GpuCountingSortModule {
    pub fn new(
        device: &wgpu::Device,
        values_buffer: &wgpu::Buffer,
        count_buffer: &wgpu::Buffer,
        workgroup_size: u32,
    ) -> Result<Self, CountingSortingError> {
        if !count_buffer.usage().contains(wgpu::BufferUsages::COPY_DST) {
            return Err(CountingSortingError::MissingBufferUsage(wgpu::BufferUsages::COPY_DST, "Count buffer"));
        }

        if !values_buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(CountingSortingError::MissingBufferUsage(wgpu::BufferUsages::STORAGE, "Values buffer"));
        }

        if !count_buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(CountingSortingError::MissingBufferUsage(wgpu::BufferUsages::STORAGE, "Count buffer"));
        }

        let count_buffer_size = count_buffer.size();
        let size: u32 = (count_buffer_size / std::mem::size_of::<u32>() as u64) as _;

        if size > workgroup_size * workgroup_size {
            return Err(CountingSortingError::SizeError(size, workgroup_size));
        }

        let sorting_id_buffer = buffers::create_buffer_for_size(
            device,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            Some("sorting id buffer"),
            count_buffer.size(),
        );

        // init bind groups
        let single_read_write_storage_buffer_bind_group_layout_with_desc = binding_builder::BindGroupLayoutBuilder::new()
            .add_binding_compute(wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            })
            .create(device, None);

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

        let counting_bind_group = binding_builder::BindGroupBuilder::new(&read_write_bind_group_layout_with_desc)
            .resource(values_buffer.as_entire_binding())
            .resource(count_buffer.as_entire_binding())
            .create(device, Some("counting_bind_group"));

        let sorting_bind_group = binding_builder::BindGroupBuilder::new(&single_read_write_storage_buffer_bind_group_layout_with_desc)
            .resource(sorting_id_buffer.as_entire_binding())
            .create(device, Some("sorting_bind_group"));

        let count_buffer_bind_group = binding_builder::BindGroupBuilder::new(&single_read_write_storage_buffer_bind_group_layout_with_desc)
            .resource(count_buffer.as_entire_binding())
            .create(device, Some("count_buffer_bind_group"));

        // Pipelines
        let counting_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("counting shader"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(
                ShaderComposer::new(include_str!("../shaders/counting.wgsl"), Some("counting"))
                    .with_shader_define("WORKGROUP_SIZE", workgroup_size.into())
                    .build()
                    .unwrap(),
            )),
        });

        let mut scan_shader_composer =
            ShaderComposer::new(include_str!("../shaders/scan.wgsl"), Some("scan")).with_shader_define("WORKGROUP_SIZE", workgroup_size.into());

        scan_shader_composer.add_shader_define("SCAN_LEVEL", 0u32.into());
        let scan_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan shader"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(scan_shader_composer.build_ref().unwrap())),
        });

        // Unable to use push_constant as it's not available in wgpu yet so we have to use a shader define for the scan level
        // Otherwise we could have used a uniform buffer to pass the scan level but this force use to submit the queue for each scan level
        scan_shader_composer.add_shader_define("SCAN_LEVEL", 1u32.into());
        let scan_level1_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan shader"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(scan_shader_composer.build().unwrap())),
        });

        let sorting_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("counting shader"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(
                ShaderComposer::new(include_str!("../shaders/sorting.wgsl"), Some("sorting"))
                    .with_shader_define("WORKGROUP_SIZE", workgroup_size.into())
                    .build()
                    .unwrap(),
            )),
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

        Ok(Self {
            workgroup_size,
            size,

            sorting_id_buffer,

            counting_bind_group,
            sorting_bind_group,
            count_buffer_bind_group,

            counting_pipeline,
            workgroup_scan_pipeline,
            workgroup_scan_level1_pipeline,
            workgroup_propagate_pipeline,
            sorting_pipeline,
        })
    }
}

impl GpuCountingSortModule {
    // TODO: find a way to store some kind of reference to the buffer to avoid the need to pass it as an argument
    pub fn dispatch_work(&self, encoder: &mut wgpu::CommandEncoder, count_buffer: &wgpu::Buffer) {
        let workgroup_size_x = (self.size + self.workgroup_size - 1u32) / self.workgroup_size;

        log::trace!("[GpuCountingSortModule] Dispatching {} workgroups of size {} (for buffer of {})", workgroup_size_x, self.workgroup_size, self.size);

        encoder.push_debug_group("Counting Sort");
        encoder.clear_buffer(count_buffer, 0, None);

        {
            let count_pass = &mut encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compting Pass"),
                timestamp_writes: None,
            });

            count_pass.set_pipeline(&self.counting_pipeline);
            count_pass.set_bind_group(0, &self.counting_bind_group, &[]);
            count_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
        }

        {
            let scan_pass = &mut encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scan Pass"),
                timestamp_writes: None,
            });

            scan_pass.set_bind_group(0, &self.count_buffer_bind_group, &[]);

            scan_pass.push_debug_group(format!("First Scan ({} workgroups)", workgroup_size_x).as_str());
            scan_pass.set_pipeline(&self.workgroup_scan_pipeline);
            scan_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
            scan_pass.pop_debug_group();
            let second_scan_workgroup_size_x = (workgroup_size_x + self.workgroup_size - 1u32) / self.workgroup_size;
            scan_pass.push_debug_group(format!("Second Scan ({} workgroups)", second_scan_workgroup_size_x).as_str());
            scan_pass.set_pipeline(&self.workgroup_scan_level1_pipeline);
            scan_pass.dispatch_workgroups(second_scan_workgroup_size_x, 1, 1);
            scan_pass.pop_debug_group();
            scan_pass.push_debug_group("Propagate");
            scan_pass.set_pipeline(&self.workgroup_propagate_pipeline);
            scan_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
            scan_pass.pop_debug_group();
        }

        {
            let sort_pass = &mut encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sort Pass"),
                timestamp_writes: None,
            });

            sort_pass.set_pipeline(&self.sorting_pipeline);
            sort_pass.set_bind_group(0, &self.counting_bind_group, &[]);
            sort_pass.set_bind_group(1, &self.sorting_bind_group, &[]);
            sort_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
        }
        encoder.pop_debug_group();
    }

    pub fn sorting_id_buffer(&self) -> &wgpu::Buffer { &self.sorting_id_buffer }
}
