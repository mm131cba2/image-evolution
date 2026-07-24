// WGSL シェーダー文字列。CGL ステップは CPU 参照 src/engine/cglStep.ts を 1:1 で移植。
// 複素場 ψ は array<vec2<f32>>（x=Re, y=Im）。境界は壁（反射＝インデックス clamp）。

// Params レイアウト（32 バイト・cglGPU.ts の writeBuffer と一致させること）:
//   L:u32, b:f32, c:f32, D:f32, dt:f32, ampRef:f32, _p0:f32, _p1:f32
const PARAMS_STRUCT = `
struct Params {
  L: u32,
  b: f32,
  c: f32,
  D: f32,
  dt: f32,
  ampRef: f32,
  _p0: f32,
  _p1: f32,
};`;

// CGL 1 ステップ（実空間陽解法・壁反射ラプラシアン）。
export const CGL_STEP_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> inBuf: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outBuf: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> P: Params;

fn at(x: i32, y: i32, L: i32) -> vec2<f32> {
  let xc = clamp(x, 0, L - 1);   // 壁=反射（端は自分を複製＝勾配ゼロ）
  let yc = clamp(y, 0, L - 1);
  return inBuf[u32(yc * L + xc)];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = i32(P.L);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= L || y >= L) { return; }
  let c = u32(y * L + x);
  let psi = inBuf[c];
  let lap = at(x - 1, y, L) + at(x + 1, y, L) + at(x, y - 1, L) + at(x, y + 1, L) - 4.0 * psi;
  let r = psi.x;
  let m = psi.y;
  let m2 = r * r + m * m;
  // (1+ib)·D·∇²ψ
  let diffRe = P.D * (lap.x - P.b * lap.y);
  let diffIm = P.D * (lap.y + P.b * lap.x);
  // −(1+ic)·|ψ|²·ψ
  let nlRe = -m2 * (r - P.c * m);
  let nlIm = -m2 * (m + P.c * r);
  outBuf[c] = vec2<f32>(
    r + P.dt * (r + diffRe + nlRe),
    m + P.dt * (m + diffIm + nlIm),
  );
}
`;

// 表示（モード B）: 位相→色相（OKLCh 循環 LUT）・振幅→彩度（芯 ψ=0 は中立色）。
export const DISPLAY_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> field: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> P: Params;
@group(0) @binding(2) var lut: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let L = i32(P.L);
  let x = clamp(i32(fc.x), 0, L - 1);
  let y = clamp(i32(fc.y), 0, L - 1);
  let psi = field[u32(y * L + x)];
  let amp = length(psi);
  let phase = atan2(psi.y, psi.x);            // [-π, π]
  let u = phase / (2.0 * 3.14159265) + 0.5;   // [0, 1)
  let col = textureSample(lut, samp, vec2<f32>(u, 0.5)).rgb;
  let ampN = clamp(amp / P.ampRef, 0.0, 1.0);
  let outc = mix(vec3<f32>(0.5), col, ampN);  // 芯(振幅0)は中立灰
  return vec4<f32>(outc, 1.0);
}
`;
