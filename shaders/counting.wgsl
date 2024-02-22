@group(0) @binding(0) var<storage, read> values : array<u32>;
@group(0) @binding(1) var<storage, read_write> counting : array<atomic<u32>>;

// from : https://github.com/JMS55/bevy/blob/solari3/crates/bevy_pbr/src/solari/global_illumination/utils.wgsl#L8-L36
fn rand_u(seed: u32) -> u32 {
    var h  = seed * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

@compute @workgroup_size(#WORKGROUP_SIZE)
fn count(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let total = arrayLength(&values);
    let index: u32 = GlobalInvocationID.x;
    if (index >= total) { return; }

    atomicAdd(&counting[values[index]], 1u);
}

@compute @workgroup_size(#WORKGROUP_SIZE)
fn reset(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let total = arrayLength(&counting);
    let index: u32 = GlobalInvocationID.x;
    if (index >= total) { return; }

    atomicStore(&counting[index], 0u);
}