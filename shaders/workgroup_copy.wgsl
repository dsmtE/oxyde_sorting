@group(0) @binding(0) var<storage, read> source : array<u32>;
@group(0) @binding(1) var<storage, read_write> destination : array<u32>;

@compute @workgroup_size(#WORKGROUP_SIZE)
fn workgroup_copy(
    // @builtin(global_invocation_id) globalInvocationId : vec3<u32>,
    @builtin(workgroup_id) workgroupId : vec3<u32>,
) {
    let wid = workgroupId.x;

    if (wid > arrayLength(&destination)) {
        return;
    }
    destination[wid] = source[(wid+1u) * #WORKGROUP_SIZE - 1u];
}
