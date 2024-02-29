@group(0) @binding(0) var<storage, read_write> values : array<u32>;

@group(1) @binding(0) var<storage, read_write> propagate_scan_values : array<u32>;

// TODO: understand how to use subgroups (https://github.com/gfx-rs/wgpu/pull/4190)
// Test reduce then scan algorithm

// Interesting links:
// https://github.com/b0nes164/GPUPrefixSums
// https://lemasyma.github.io/cours/posts/irgpu_patterns/#scan-pattern-at-the-block-or-grid-level
// https://rd.yyrcd.com/CUDA/2022-03-14-Single-pass%20Parallel%20Prefix%20Scan%20with%20Decoupled%20Look-back.pdf

// Should be implemented using deviceMemoryBarrier but it's not available in wgpu yet
// https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html

@compute @workgroup_size(#WORKGROUP_SIZE)
fn workgroup_scan(
    @builtin(global_invocation_id) globalInvocationId : vec3<u32>,
    @builtin(local_invocation_id) localInvocationId : vec3<u32>,
    // @builtin(workgroup_id) workgroupId : vec3<u32>,
    // @builtin(num_workgroups) numWorkgroups : vec3<u32>,
) {
    let total = arrayLength(&values);

    let gid: u32 = globalInvocationId.x;
    let lid: u32 = localInvocationId.x;
    // let wid: u32 = workgroupId.x;
    // gid = wid * numWorkgroups.x + lid;

    if (gid >= total) { return; }

    // do the scan at the workgroup level using workgroup memory barrier
    // Kogge-Stone method
    for (var offset = 1u; offset < #WORKGROUP_SIZE; offset <<= 1u) {
        workgroupBarrier();
        if (lid >= offset) {
            let temp = values[gid - offset];
            workgroupBarrier();
            values[gid] += temp;
        }
    }
    // TODO
    //  Do the scan on the last element of each workgroup stored in a separate buffer (to be able to sort it wihin one workgroup as deviceMemoryBarrier is not available yet in wgpu)
    // Then reduce and sum over all the workgroups to get the global scan 
}

@compute @workgroup_size(#WORKGROUP_SIZE)
fn workgroup_propagate(
    @builtin(global_invocation_id) globalInvocationId : vec3<u32>,
    @builtin(workgroup_id) workgroupId : vec3<u32>,
) {
    let total = arrayLength(&values);

    let gid: u32 = globalInvocationId.x;
    let wid: u32 = workgroupId.x;

    if (gid >= total) { return; }
    if (wid == 0) { return; }

    values[gid] += propagate_scan_values[wid-1];
}
