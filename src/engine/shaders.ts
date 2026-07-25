// WGSL シェーダー。CGL ステップは CPU 参照 src/engine/cglStep.ts を 1:1 で移植。
// 複素場 ψ は array<vec2<f32>>（x=Re, y=Im）。境界は壁（反射＝インデックス clamp）。
//
// Params レイアウト（48 バイト・cglGPU.ts の writeBuffer と一致させること）:
//   0:L(u32) 4:b 8:c 12:D 16:dt 20:ampRef 24:speed 28:mode(u32) 32:blend 36:phaseRef 40:dynamics(u32) 44:pad
const PARAMS_STRUCT = `
struct Params {
  L: u32, b: f32, c: f32, D: f32,
  dt: f32, ampRef: f32, speed: f32, mode: u32,
  blend: f32, phaseRef: f32, dynamics: u32, _p2: f32,
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

// Gray-Scott 反応拡散（re=u, im=v ∈ [0,1]）。CPU 参照 dynamics.ts grayScottStep と 1:1。
export const GRAYSCOTT_WGSL = /* wgsl */ `
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
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= L || y >= L) { return; }
  let c = u32(y * L + x);
  let s = inBuf[c];
  let lap = at(x-1,y,L) + at(x+1,y,L) + at(x,y-1,L) + at(x,y+1,L) - 4.0 * s;
  let u = s.x; let v = s.y;
  let uvv = u * v * v;
  let Du = 0.16; let Dv = 0.08; let F = 0.037; let k = 0.06; let dt = 1.0;
  outBuf[c] = vec2<f32>(
    u + dt * (Du * lap.x - uvv + F * (1.0 - u)),
    v + dt * (Dv * lap.y + uvv - (F + k) * v),
  );
}
`;

// Asymptotic Lenia（re=u ∈ [0,1]・指数オイラー・自己正規化ガウス殻カーネル）。
// CPU 参照 dynamics.ts leniaStep と 1:1（mu,sigma,R,kr0,kw,dt を一致させること）。
export const LENIA_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> inBuf: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outBuf: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> P: Params;
fn uAt(x: i32, y: i32, L: i32) -> f32 {
  return inBuf[u32(clamp(y, 0, L - 1) * L + clamp(x, 0, L - 1))].x;
}
fn bell(x: f32, m: f32, s: f32) -> f32 { let d = (x - m) / s; return exp(-0.5 * d * d); }
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = i32(P.L);
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= L || y >= L) { return; }
  let R = 8; let mu = 0.15; let sigma = 0.017; let kr0 = 0.5; let kw = 0.15; let dt = 0.3;
  var accWU = 0.0; var accW = 0.0;
  for (var dy = -R; dy <= R; dy = dy + 1) {
    for (var dx = -R; dx <= R; dx = dx + 1) {
      let r = sqrt(f32(dx*dx + dy*dy)) / f32(R);
      if (r > 1.0 || r == 0.0) { continue; }
      let w = bell(r, kr0, kw);
      accWU = accWU + w * uAt(x + dx, y + dy, L);
      accW = accW + w;
    }
  }
  let pot = select(0.0, accWU / accW, accW > 0.0);
  let g = bell(pot, mu, sigma);
  let decay = exp(-dt);
  let u = inBuf[u32(y * L + x)].x;
  outBuf[u32(y * L + x)] = vec2<f32>(decay * u + (1.0 - decay) * g, 0.0);
}
`;

// 色差拡散（re=Cb, im=Cr を D·dt·∇² で拡散・輝度 Y は状態に持たない）。
// CPU 参照 dynamics.ts chromaStep と 1:1。
export const CHROMA_WGSL = /* wgsl */ `
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
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= L || y >= L) { return; }
  let c = u32(y * L + x);
  let s = inBuf[c];
  let lap = at(x-1,y,L) + at(x+1,y,L) + at(x,y-1,L) + at(x,y+1,L) - 4.0 * s;
  outBuf[c] = s + (P.D * P.dt) * lap;
}
`;

// 四元数 CGL（全色発展）。状態 q=vec4(w,x,y,z)。CPU 参照 dynamics.ts quatCglStep と 1:1。
// I=(1,1,1)/√3（軸）。∂q/∂t = q + (1+bI)D∇²q − (1+cI)|q|²q。
export const QUAT_STEP_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> inBuf: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outBuf: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> P: Params;
fn at(x: i32, y: i32, L: i32) -> vec4<f32> {
  return inBuf[u32(clamp(y, 0, L - 1) * L + clamp(x, 0, L - 1))];
}
// 四元数積 a⊗b（a=(aw,ax,ay,az)）。
fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
    a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
    a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
    a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x,
  );
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = i32(P.L);
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= L || y >= L) { return; }
  let c = u32(y * L + x);
  let q = inBuf[c];  // (w,x,y,z)
  let lap = at(x-1,y,L) + at(x+1,y,L) + at(x,y-1,L) + at(x,y+1,L) - 4.0 * q;
  let m2 = dot(q, q);
  let s = 0.57735027; // 1/√3
  let Ab = vec4<f32>(1.0, P.b*s, P.b*s, P.b*s);
  let Ac = vec4<f32>(1.0, P.c*s, P.c*s, P.c*s);
  // qmul は a=(aw,ax,ay,az) 規約。ここでは vec4 の .x=w,.y=x,.z=y,.w=z として扱う。
  let diff = qmul(Ab, P.D * lap);
  let nl = qmul(Ac, m2 * q);
  outBuf[c] = q + P.dt * (q + diff - nl);
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

// MacCormack 第2パス: φ1(前進再移流)で誤差補正し、逆写像を鮮鋭に保つ（写真がボケない）。
// mapOut = φ1 + 0.5·(M − advect_fwd(φ1))、リミッタで M の近傍範囲にクランプ（オーバーシュート防止）。
export const ADVECT_MC2_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read> phi1: array<vec2<f32>>;   // 第1パス結果
@group(0) @binding(1) var<storage, read> M: array<vec2<f32>>;      // 元の逆写像
@group(0) @binding(2) var<storage, read_write> outMap: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> psi: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> P: Params;

fn ix(x: i32, y: i32, L: i32) -> u32 { return u32(clamp(y, 0, L - 1) * L + clamp(x, 0, L - 1)); }
fn imAt(x: i32, y: i32, L: i32) -> f32 { return psi[ix(x, y, L)].y; }

fn samplePhi(p: vec2<f32>, L: i32) -> vec2<f32> {
  let x = clamp(p.x, 0.0, f32(L - 1));
  let y = clamp(p.y, 0.0, f32(L - 1));
  let x0 = i32(floor(x));
  let y0 = i32(floor(y));
  let x1 = min(x0 + 1, L - 1);
  let y1 = min(y0 + 1, L - 1);
  let fx = x - f32(x0);
  let fy = y - f32(y0);
  let a = phi1[u32(y0 * L + x0)];
  let b = phi1[u32(y0 * L + x1)];
  let cc = phi1[u32(y1 * L + x0)];
  let d = phi1[u32(y1 * L + x1)];
  return mix(mix(a, b, fx), mix(cc, d, fx), fy);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let L = i32(P.L);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= L || y >= L) { return; }
  let c = u32(y * L + x);
  let dgdx = 0.5 * (imAt(x + 1, y, L) - imAt(x - 1, y, L));
  let dgdy = 0.5 * (imAt(x, y + 1, L) - imAt(x, y - 1, L));
  let v = P.speed * vec2<f32>(dgdy, -dgdx);
  let p = vec2<f32>(f32(x), f32(y));
  let phi2 = samplePhi(p + v, L);            // φ1 を前進再移流 ≈ M のはず
  let corrected = phi1[c] + 0.5 * (M[c] - phi2);
  // リミッタ: M の (x−v) 近傍 4 隅の範囲へクランプ
  let bx0 = i32(floor(p.x - v.x));
  let by0 = i32(floor(p.y - v.y));
  let m00 = M[ix(bx0, by0, L)];
  let m10 = M[ix(bx0 + 1, by0, L)];
  let m01 = M[ix(bx0, by0 + 1, L)];
  let m11 = M[ix(bx0 + 1, by0 + 1, L)];
  let lo = min(min(m00, m10), min(m01, m11));
  let hi = max(max(m00, m10), max(m01, m11));
  outMap[c] = clamp(corrected, lo, hi);
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
@group(0) @binding(7) var<storage, read> quatField: array<vec4<f32>>; // 四元数場 q=(w,x,y,z)

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

// BT.601 YCbCr→sRGB（dynamics.ts yCbCrToRgb と一致）。
fn ycbcr2rgb(Y: f32, Cb: f32, Cr: f32) -> vec3<f32> {
  return vec3<f32>(Y + 1.402 * Cr, Y - 0.344136 * Cb - 0.714136 * Cr, Y + 1.772 * Cb);
}

// OKLab (L,a,b) → 線形 sRGB（color.ts oklabToLinear と同一係数・pow でなく積で3乗）。
fn oklab2lin(L: f32, a: f32, b: f32) -> vec3<f32> {
  let l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  let m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  let s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  let l = l_ * l_ * l_; let m = m_ * m_ * m_; let s = s_ * s_ * s_;
  return vec3<f32>(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  );
}
fn lin2srgb1(x: f32) -> f32 {
  let c = clamp(x, 0.0, 1.0);
  return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
}
// 四元数積（DISPLAY 用・dynamics.ts qmul と同一規約 .x=w,.y=x,.z=y,.w=z）。
fn qmulD(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
    a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
    a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
    a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x,
  );
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
  // 共回転フレーム: CGL の一様位相回転(dθ/dt=−c)を打ち消して色の全画面ストロボを止める。
  // phaseRef=0 なら従来どおり回る。
  let phase = atan2(psi.y, psi.x) + P.phaseRef;
  // u=fract(phase/2π): LUT 色相=2π·u なので表示色相=arg ψ に一致（種 color の色を保つ）。
  // +0.5 offset は色相を π ずらす＝補色になる（checks/color-field.py で確認）ので使わない。
  let u = fract(phase / (2.0 * 3.14159265));
  let lutc = textureSample(lut, samp, vec2<f32>(u, 0.5)).rgb;
  // 振幅→彩度・位相→色相。種 ψ=(a,b)（OKLab の対向色平面）と同じ規約なので、
  // t=0 では写真の色相・彩度がそのまま出る。ψ=0（無彩色・渦の芯）は中立灰。
  let bcol = mix(vec3<f32>(0.5), lutc, clamp(amp / P.ampRef, 0.0, 1.0));

  // A: 逆写像で原本をサンプル（+0.5 でテクセル中心に合わせる＝半texelずれ防止）
  let src = (fmap[c] + vec2<f32>(0.5)) / f32(L);
  let acol = textureSample(orig, origSamp, src).rgb;
  // chroma 用: 原本を同位置（恒等）でサンプルし輝度 Y を取る（移流しない）。
  let idc = (vec2<f32>(f32(x), f32(y)) + vec2<f32>(0.5)) / f32(L);
  let origHere = textureSample(orig, origSamp, idc).rgb;

  var outc: vec3<f32>;
  if (P.dynamics == 1u) {
    // Gray-Scott: 写真の色を v(=state.y) の強度で明暗変調（写真ごとに色が変わる・真っ黒回避）
    let fi = clamp(psi.y / 0.3, 0.0, 1.0);
    outc = clamp(mix(origHere * 0.12, origHere * 1.5, fi), vec3<f32>(0.0), vec3<f32>(1.0));
  } else if (P.dynamics == 2u) {
    // Lenia: 写真の色を u(=state.x) で明暗変調（黄色一色でなく写真依存の色に）
    let fi = clamp(psi.x, 0.0, 1.0);
    outc = clamp(mix(origHere * 0.12, origHere * 1.5, fi), vec3<f32>(0.0), vec3<f32>(1.0));
  } else if (P.dynamics == 3u) {
    // 色差拡散: 輝度は原本・色差は発展した Cb,Cr（写真の形は保ち色だけ滲む）
    let Y = 0.299 * origHere.r + 0.587 * origHere.g + 0.114 * origHere.b;
    outc = clamp(ycbcr2rgb(Y, psi.x, psi.y), vec3<f32>(0.0), vec3<f32>(1.0));
  } else if (P.dynamics == 4u) {
    // 四元数（全色発展）: 純虚部(x,y,z)→OKLab(L,a,b)。共回転で均質スピンを打ち消す。
    var q = quatField[c];
    let s3 = 0.57735027; // 1/√3（軸 I=(1,1,1)/√3）
    // 共回転: exp(phaseRef·I)⊗q で均質左回転を相殺（phaseRef=0 なら回る）。
    let rot = vec4<f32>(cos(P.phaseRef), sin(P.phaseRef) * s3, sin(P.phaseRef) * s3, sin(P.phaseRef) * s3);
    q = qmulD(rot, q);
    let Lab = vec3<f32>(0.5 + 0.5 * q.y, 0.3 * q.z, 0.3 * q.w); // Im→OKLab（種の逆変換）
    let lin = oklab2lin(clamp(Lab.x, 0.0, 1.0), Lab.y, Lab.z);
    outc = vec3<f32>(lin2srgb1(lin.r), lin2srgb1(lin.g), lin2srgb1(lin.b));
  } else if (P.mode == 1u) { outc = bcol; }
  else if (P.mode == 0u) { outc = acol; }
  else { outc = mix(acol, bcol, P.blend); }
  return vec4<f32>(outc, 1.0);
}
`;
