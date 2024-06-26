struct InitUniforms {
    current_time_ms: u32,
    init_method: u32,
    init_value: u32,
    max_value: u32,
}

@group(0) @binding(0) var<storage, read_write> values : array<u32>;
@group(1) @binding(0) var<uniform> uniforms : InitUniforms;

// from : https://github.com/JMS55/bevy/blob/solari3/crates/bevy_pbr/src/solari/global_illumination/utils.wgsl#L8-L36
fn rand_u(seed: u32) -> u32 {
    var h  = seed * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

@compute @workgroup_size(#WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let total = arrayLength(&values);
    let index: u32 = GlobalInvocationID.x;
    if (index >= total) { return; }

    if (uniforms.init_method == 0u) {
        values[index] = uniforms.init_value % uniforms.max_value;
    }else if (uniforms.init_method == 1u) {
        values[index] = index % uniforms.max_value;
    } else {
        values[index] = rand_u(index + uniforms.current_time_ms) % uniforms.max_value;
    }
}