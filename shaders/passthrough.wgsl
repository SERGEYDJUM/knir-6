@group(0) @binding(0) var r_color: texture_2d<f32>;
@group(0) @binding(1) var r_sampler: sampler;

@fragment fn main(@builtin(position) _sv_position: vec4<f32>, @location(0) coords: vec2<f32>) -> @location(0) vec4<f32> {
    let texel = textureSample(r_color, r_sampler, coords);
    return vec4f(texel.b, texel.g, texel.r, texel.w);
}