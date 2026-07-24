// WGSL シェーダー。CGL ステップは CPU 参照 src/engine/cglStep.ts を 1:1 で移植。
// 複素場 ψ は array<vec2<f32>>（x=Re, y=Im）。境界は壁（反射＝インデックス clamp）。
//
// Params レイアウト（48 バイト・cglGPU.ts の writeBuffer と一致させること）:
//   0:L(u32) 4:b 8:c 12:D 16:dt 20:ampRef 24:speed 28:mode(u32) 32:blend 36..:pad
const PARAMS_STRUCT = `
struct Params {
  L: u32, b: f32, c: f32, D: f32,
  dt: f32, ampRef: f32, speed: f32, mode: u32,
  blend: f32, _p0: f32, _p1: f32, _p2: f32,
};`;

// CGL 1 ステップ（実空間陽解法・壁反射ラプラシアン）。
export const CGL_STEP_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> inBuf: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outBuf: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> P: Params;

fn at(x: i32, y: i32, L: i32) -> vec2<f32> {
  return inBuf[u32(clamp(y, 0, L - 1) * L + clamp(x, 0, L - 1))];
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
  let diffRe = P.D * (lap.x - P.b * lap.y);
  let diffIm = P.D * (lap.y + P.b * lap.x);
  let nlRe = -m2 * (r - P.c * m);
  let nlIm = -m2 * (m + P.c * r);
  outBuf[c] = vec2<f32>(
    r + P.dt * (r + diffRe + nlRe),
    m + P.dt * (m + diffIm + nlIm),
  );
}
`;

// モード A の移流: 逆写像（原本をどこから引くか）を ψ 由来の速度で 1 ステップ後退。
// v = speed·∇⊥Im(ψ)（非圧縮・非往復）。v1 は単純 semi-Lagrangian（MacCormack 上位互換は後日）。
export const ADVECT_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> mapIn: array<vec2<f32>>;   // 逆写像(ピクセル座標)
@group(0) @binding(1) var<storage, read_write> mapOut: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> psi: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> P: Params;

fn imAt(x: i32, y: i32, L: i32) -> f32 {
  return psi[u32(clamp(y, 0, L - 1) * L + clamp(x, 0, L - 1))].y;
}

// 逆写像バッファのバイリニア標本（端は clamp）。
fn sampleMap(p: vec2<f32>, L: i32) -> vec2<f32> {
  let x = clamp(p.x, 0.0, f32(L - 1));
  let y = clamp(p.y, 0.0, f32(L - 1));
  let x0 = i32(floor(x));
  let y0 = i32(floor(y));
  let x1 = min(x0 + 1, L - 1);
  let y1 = min(y0 + 1, L - 1);
  let fx = x - f32(x0);
  let fy = y - f32(y0);
  let a = mapIn[u32(y0 * L + x0)];
  let b = mapIn[u32(y0 * L + x1)];
  let cc = mapIn[u32(y1 * L + x0)];
  let d = mapIn[u32(y1 * L + x1)];
  return mix(mix(a, b, fx), mix(cc, d, fx), fy);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = i32(P.L);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= L || y >= L) { return; }
  // v = speed·∇⊥Im(ψ) = speed·(∂g/∂y, −∂g/∂x)
  let dgdx = 0.5 * (imAt(x + 1, y, L) - imAt(x - 1, y, L));
  let dgdy = 0.5 * (imAt(x, y + 1, L) - imAt(x, y - 1, L));
  let v = P.speed * vec2<f32>(dgdy, -dgdx);
  let src = vec2<f32>(f32(x), f32(y)) - v;   // 後退トレース
  mapOut[u32(y * L + x)] = sampleMap(src, L);
}
`;

// 表示: mode 0=A(原本を逆写像でサンプル) / 1=B(ψ→OKLCh) / 2=blend。
export const DISPLAY_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> field: array<vec2<f32>>;   // ψ
@group(0) @binding(1) var<uniform> P: Params;
@group(0) @binding(2) var lut: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<storage, read> fmap: array<vec2<f32>>;    // 逆写像
@group(0) @binding(5) var orig: texture_2d<f32>;                    // 原本(sRGB)
@group(0) @binding(6) var origSamp: sampler;

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
  let c = u32(y * L + x);

  // B: ψ → OKLCh
  let psi = field[c];
  let amp = length(psi);
  let phase = atan2(psi.y, psi.x);
  let u = phase / (2.0 * 3.14159265) + 0.5;
  let lutc = textureSample(lut, samp, vec2<f32>(u, 0.5)).rgb;
  let bcol = mix(vec3<f32>(0.5), lutc, clamp(amp / P.ampRef, 0.0, 1.0));

  // A: 逆写像で原本をサンプル
  let src = fmap[c] / f32(L);
  let acol = textureSample(orig, origSamp, src).rgb;

  var outc: vec3<f32>;
  if (P.mode == 1u) { outc = bcol; }
  else if (P.mode == 0u) { outc = acol; }
  else { outc = mix(acol, bcol, P.blend); }
  return vec4<f32>(outc, 1.0);
}
`;
