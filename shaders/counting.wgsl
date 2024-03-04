@group(0) @binding(0) var<storage, read> values : array<u32>;
@group(0) @binding(1) var<storage, read_write> counting : array<atomic<u32>>;

@compute @workgroup_size(#WORKGROUP_SIZE)
fn count(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let total = arrayLength(&values);
    let index: u32 = GlobalInvocationID.x;
    if (index >= total) { return; }

    atomicAdd(&counting[values[index]], 1u);
}