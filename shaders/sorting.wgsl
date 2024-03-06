@group(0) @binding(0) var<storage, read> values : array<u32>;
@group(0) @binding(1) var<storage, read_write> counting : array<atomic<u32>>;

@group(1) @binding(0) var<storage, read_write> sorting_idx : array<u32>;

@compute @workgroup_size(#WORKGROUP_SIZE)
// counting sort (not stable as done in parallel)
fn sort (@builtin(global_invocation_id) globalInvocationId : vec3<u32>) {
    let total = arrayLength(&values);
    let gid: u32 = globalInvocationId.x;

    if (gid >= total) { return; }

    let value = values[gid];
    let count = atomicSub(&counting[value], 1u);
    sorting_idx[count-1u] = gid;
}