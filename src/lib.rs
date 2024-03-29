use log;

use oxyde::{
    anyhow::Result,
    wgpu,
    wgpu_utils::{binding_builder, buffers, ShaderComposer}
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
    value_size: u32,
    count_size: u32,

    sorting_id_buffer: wgpu::Buffer,

    counting_bind_group: wgpu::BindGroup,
    sorting_bind_group: wgpu::BindGroup,
    count_buffer_bind_group: wgpu::BindGroup,

    counting_pipeline: wgpu::ComputePipeline,

    workgroup_scan_pipelines: Vec<wgpu::ComputePipeline>,
    workgroup_propagate_pipelines: Vec<wgpu::ComputePipeline>,
    sorting_pipeline: wgpu::ComputePipeline,
}

#[derive(Debug)]
pub enum CountingSortingError {
    MissingBufferUsage(wgpu::BufferUsages, &'static str),
    ToManyScanThenPropagateLevels(u32, u32, u32),
}

impl std::fmt::Display for CountingSortingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CountingSortingError::MissingBufferUsage(buffer_usage, buffer_name) =>
                write!(f, "Missing buffer usage {:?} for {}", buffer_usage, buffer_name),
            CountingSortingError::ToManyScanThenPropagateLevels(size, workgroup_size, scan_then_propagate_levels) => {
                write!(
                    f,
                    "Unable to handle a buffer of size {} with a workgroup size of {}, this require too many scan and propagate levels ({} levels)",
                    size,
                    workgroup_size,
                    scan_then_propagate_levels
                )
            }
        }
    }
}

impl std::error::Error for CountingSortingError {}

//This function is used to compute the number of scan then propagate levels required to scan the count buffer for a given size and workgroup size
fn scan_then_propagate_level_count(size: u32, workgroup_size: u32) -> u32 {
    let mut count = 1;
    let mut temp_size = size / workgroup_size;
    while temp_size > 0 {
        count += 1;
        temp_size /= workgroup_size;
    }
    count
}

fn workgroup_size_per_level(size: u32, workgroup_size: u32, level: u32) -> Vec<u32> {
    std::iter::successors(
        Some(size),
        |&x| Some((x + workgroup_size - 1) / workgroup_size))
    .take((level+1) as usize)
    .skip(1)
    .collect()
}

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

        let count_size: u32 = (count_buffer.size() / std::mem::size_of::<u32>() as u64) as _;
        let value_size = (values_buffer.size() / std::mem::size_of::<u32>() as u64) as _;

        let scan_then_propagate_level_count = scan_then_propagate_level_count(count_size, workgroup_size);

        if scan_then_propagate_level_count > 4 {
            return Err(CountingSortingError::ToManyScanThenPropagateLevels(count_size, workgroup_size, scan_then_propagate_level_count));
        }

        let sorting_id_buffer = buffers::create_buffer_for_size(
            device,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            Some("sorting id buffer"),
            values_buffer.size(),
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

        
        let mut workgroup_scan_pipelines = Vec::with_capacity(scan_then_propagate_level_count as usize);
        let mut workgroup_propagate_pipelines = Vec::with_capacity((scan_then_propagate_level_count-1) as usize);

        for scan_then_propagate_level in 0..scan_then_propagate_level_count {
            // Unable to use push_constant as it's not available in wgpu yet so we have to use a shader define for the scan level and recompile the shader for each level
            // Otherwise we could have used a uniform buffer to pass the scan level but this force use to submit the queue for each scan level
            scan_shader_composer.add_shader_define("SCAN_LEVEL", scan_then_propagate_level.into());

            let scan_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scan shader"),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(scan_shader_composer.build_ref().unwrap())),
            });

            workgroup_scan_pipelines.push(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(format!("workgroup scan pipeline (level {})", scan_then_propagate_level).as_str()),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(format!("workgroup scan pipeline layout (level {})", scan_then_propagate_level).as_str()),
                    bind_group_layouts: &[&single_read_write_storage_buffer_bind_group_layout_with_desc.layout],
                    push_constant_ranges: &[],
                })),
                module: &scan_shader_module,
                entry_point: "workgroup_scan",
            }));

            if scan_then_propagate_level < scan_then_propagate_level_count - 1 {
                workgroup_propagate_pipelines.push(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(format!("workgroup propagate pipeline (level {})", scan_then_propagate_level).as_str()),
                    layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(format!("workgroup propagate pipeline layout (level {})", scan_then_propagate_level).as_str()),
                        bind_group_layouts: &[&single_read_write_storage_buffer_bind_group_layout_with_desc.layout],
                        push_constant_ranges: &[],
                    })),
                    module: &scan_shader_module,
                    entry_point: "workgroup_propagate",
                }));
            }
        }

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
            value_size,
            count_size,

            sorting_id_buffer,

            counting_bind_group,
            sorting_bind_group,
            count_buffer_bind_group,

            counting_pipeline,
            workgroup_scan_pipelines,
            workgroup_propagate_pipelines,
            sorting_pipeline,
        })
    }
}

impl GpuCountingSortModule {
    // TODO: find a way to store some kind of reference to the buffer to avoid the need to pass it as an argument
    pub fn dispatch_work(&self, encoder: &mut wgpu::CommandEncoder, count_buffer: &wgpu::Buffer) {
        log::trace!("[GpuCountingSortModule] workgroups of size {} (for value buffer of {} and counting buffer or {})", self.workgroup_size, self.value_size, self.count_size);

        let value_workgroup_size_x = (self.value_size + self.workgroup_size - 1) / self.workgroup_size;
        encoder.push_debug_group("Counting Sort");
        encoder.clear_buffer(count_buffer, 0, None);

        {
            let count_pass = &mut encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compting Pass"),
                timestamp_writes: None,
            });

            count_pass.set_pipeline(&self.counting_pipeline);
            count_pass.set_bind_group(0, &self.counting_bind_group, &[]);
            count_pass.dispatch_workgroups(value_workgroup_size_x, 1, 1);
        }

        {
            let scan_pass = &mut encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scan Pass"),
                timestamp_writes: None,
            });

            scan_pass.set_bind_group(0, &self.count_buffer_bind_group, &[]);
            
            let scan_workgroup_sizes = workgroup_size_per_level(self.count_size, self.workgroup_size, self.workgroup_scan_pipelines.len() as u32);
            
            for (workgroup_scan_pipeline, workgroup_size_x) in self.workgroup_scan_pipelines.iter().zip(scan_workgroup_sizes.iter()) {
                scan_pass.push_debug_group(format!("Scan ({} workgroups)", workgroup_size_x).as_str());
                log::trace!("[GpuCountingSortModule] Dispatching Scan ({} workgroups)", workgroup_size_x);
                scan_pass.set_pipeline(workgroup_scan_pipeline);
                scan_pass.dispatch_workgroups(*workgroup_size_x, 1, 1);
                scan_pass.pop_debug_group();
            }

            for (workgroup_propagate_pipeline, workgroup_size_x) in self.workgroup_propagate_pipelines.iter().rev().zip(scan_workgroup_sizes.iter().rev().skip(1)) {
                scan_pass.push_debug_group(format!("Propagate ({} workgroups)", workgroup_size_x).as_str());
                log::trace!("[GpuCountingSortModule] Dispatching Propagate ({} workgroups)", workgroup_size_x);
                scan_pass.set_pipeline(workgroup_propagate_pipeline);
                scan_pass.dispatch_workgroups(*workgroup_size_x, 1, 1);
                scan_pass.pop_debug_group();
            }
        }

        {
            let sort_pass = &mut encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sort Pass"),
                timestamp_writes: None,
            });

            sort_pass.set_pipeline(&self.sorting_pipeline);
            sort_pass.set_bind_group(0, &self.counting_bind_group, &[]);
            sort_pass.set_bind_group(1, &self.sorting_bind_group, &[]);
            sort_pass.dispatch_workgroups(value_workgroup_size_x, 1, 1);
        }
        encoder.pop_debug_group();
    }

    pub fn sorting_id_buffer(&self) -> &wgpu::Buffer { &self.sorting_id_buffer }
}
