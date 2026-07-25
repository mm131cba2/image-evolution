// CGL 以外の力学の CPU 参照実装（GPU/WGSL の移植元・数値の真実）。
// いずれも状態を vec2 の 2 チャンネルに載せ、既存のピンポンバッファを共有する。
// 検証: 設計ノート checks/color-rd.py（Gray-Scott の Turing／YCbCr の輝度保持）,
//       checks/memory.py・numerics.py（Lenia の指数オイラー・記憶時間）。

// 壁(Neumann=反射)5 点ラプラシアン（cglStep.ts と同じ規約）。
function lapReflect(f: Float32Array, L: number, out: Float32Array): void {
  for (let y = 0; y < L; y++) {
    const yU = y > 0 ? y - 1 : 0;
    const yD = y < L - 1 ? y + 1 : L - 1;
    for (let x = 0; x < L; x++) {
      const xL = x > 0 ? x - 1 : 0;
      const xR = x < L - 1 ? x + 1 : L - 1;
      const c = y * L + x;
      out[c] = f[y * L + xL] + f[y * L + xR] + f[yU * L + x] + f[yD * L + x] - 4 * f[c];
    }
  }
}

// ---------------------------------------------------------------------------
// Gray-Scott 反応拡散（Turing パターン）。状態 re=u, im=v ∈ [0,1]。
// du = Du∇²u − u v² + F(1−u) / dv = Dv∇²v + u v² − (F+k)v。内部 dt=1（古典値）。
// ---------------------------------------------------------------------------
export const GS = { Du: 0.16, Dv: 0.08, F: 0.037, k: 0.06, dt: 1.0 } as const;

export function grayScottStep(re: Float32Array, im: Float32Array, L: number): void {
  const n = L * L;
  const lu = new Float32Array(n);
  const lv = new Float32Array(n);
  lapReflect(re, L, lu);
  lapReflect(im, L, lv);
  const { Du, Dv, F, k, dt } = GS;
  for (let i = 0; i < n; i++) {
    const u = re[i];
    const v = im[i];
    const uvv = u * v * v;
    re[i] = u + dt * (Du * lu[i] - uvv + F * (1 - u));
    im[i] = v + dt * (Dv * lv[i] + uvv - (F + k) * v);
  }
}

// ---------------------------------------------------------------------------
// Asymptotic Lenia（連続ライフ）。状態 re=u ∈ [0,1]（im は未使用・0）。
// 指数オイラー: u' = e^{-dt}·u + (1−e^{-dt})·G(K*u)。G はベル（値域 [0,1]）。
// カーネル K はガウス殻（r0=0.5）を自己正規化（Σw で割る）。
// ---------------------------------------------------------------------------
export const LENIA = { mu: 0.15, sigma: 0.017, R: 8, kr0: 0.5, kw: 0.15, dt: 0.3 } as const;

function bell(x: number, m: number, s: number): number {
  const d = (x - m) / s;
  return Math.exp(-0.5 * d * d);
}

export function leniaStep(re: Float32Array, im: Float32Array, L: number): void {
  const n = L * L;
  const out = new Float32Array(n);
  const { mu, sigma, R, kr0, kw, dt } = LENIA;
  const decay = Math.exp(-dt);
  for (let y = 0; y < L; y++) {
    for (let x = 0; x < L; x++) {
      let accWU = 0;
      let accW = 0;
      for (let dy = -R; dy <= R; dy++) {
        const yy = Math.min(L - 1, Math.max(0, y + dy));
        for (let dx = -R; dx <= R; dx++) {
          const r = Math.hypot(dx, dy) / R;
          if (r > 1 || r === 0) continue;
          const w = bell(r, kr0, kw);
          const xx = Math.min(L - 1, Math.max(0, x + dx));
          accWU += w * re[yy * L + xx];
          accW += w;
        }
      }
      const pot = accW > 0 ? accWU / accW : 0; // 正規化ポテンシャル ∈ [0,1]
      const g = bell(pot, mu, sigma); // 成長目標 ∈ [0,1]
      const u = re[y * L + x];
      out[y * L + x] = decay * u + (1 - decay) * g;
    }
  }
  re.set(out);
  im.fill(0);
}

// ---------------------------------------------------------------------------
// 色差拡散（YCbCr）。状態 re=Cb, im=Cr。輝度 Y は状態に持たず表示で原本から取る
// ＝写真の形（輝度エッジ）は 100% 保ったまま色だけ滲む。dt·D·∇² で Cb,Cr を拡散。
// ---------------------------------------------------------------------------
export function chromaStep(re: Float32Array, im: Float32Array, L: number, D: number, dt: number): void {
  const n = L * L;
  const lb = new Float32Array(n);
  const lr = new Float32Array(n);
  lapReflect(re, L, lb);
  lapReflect(im, L, lr);
  const a = D * dt;
  for (let i = 0; i < n; i++) {
    re[i] = re[i] + a * lb[i];
    im[i] = im[i] + a * lr[i];
  }
}

// ---------------------------------------------------------------------------
// 四元数 CGL（全色発展）。状態 q=(w,x,y,z)。純虚部 (x,y,z) を色の3次元ベクトル
// (OKLab L−0.5, a, b) とみなす。w は色に使わず内部自由度として走らせる。
// ∂q/∂t = q + (1+bI)D∇²q − (1+cI)|q|²q,  I=単位純虚四元数（回転軸・傾けて全成分を動かす）。
// 複素 CGL の i を四元数 I に置換した一般化。|q|→1 のアトラクタ（checks/quaternion-color.py）。
// ---------------------------------------------------------------------------
const A = 1 / Math.sqrt(3);
export const QUAT_AXIS: readonly [number, number, number] = [A, A, A]; // I の純虚部（傾けた軸）

// 四元数積 (aw,ax,ay,az)⊗(bw,bx,by,bz)。
function qmul(
  aw: number, ax: number, ay: number, az: number,
  bw: number, bx: number, by: number, bz: number,
): [number, number, number, number] {
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

// 4 成分を interleave した Float32Array（q[i*4+0..3]=w,x,y,z）を 1 ステップ進める。
export function quatCglStep(q: Float32Array, L: number, p: { b: number; c: number; D: number; dt: number }): void {
  const n = L * L;
  const { b, c, D, dt } = p;
  const [Ix, Iy, Iz] = QUAT_AXIS;
  // 各成分のラプラシアン（壁反射）。
  const comp = new Float32Array(n);
  const lap = [new Float32Array(n), new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  for (let k = 0; k < 4; k++) {
    for (let i = 0; i < n; i++) comp[i] = q[i * 4 + k];
    lapReflect(comp, L, lap[k]);
  }
  for (let i = 0; i < n; i++) {
    const w = q[i * 4], x = q[i * 4 + 1], y = q[i * 4 + 2], z = q[i * 4 + 3];
    const m2 = w * w + x * x + y * y + z * z;
    const lw = lap[0][i], lx = lap[1][i], ly = lap[2][i], lz = lap[3][i];
    // (1+bI)⊗(D·∇²q)
    const [dw, dx, dy, dz] = qmul(1, b * Ix, b * Iy, b * Iz, D * lw, D * lx, D * ly, D * lz);
    // (1+cI)⊗(|q|²q)
    const [nw, nx, ny, nz] = qmul(1, c * Ix, c * Iy, c * Iz, m2 * w, m2 * x, m2 * y, m2 * z);
    q[i * 4] = w + dt * (w + dw - nw);
    q[i * 4 + 1] = x + dt * (x + dx - nx);
    q[i * 4 + 2] = y + dt * (y + dy - ny);
    q[i * 4 + 3] = z + dt * (z + dz - nz);
  }
}

// BT.601（フルレンジ）sRGB ↔ YCbCr。Cb,Cr は [−0.5,0.5]。
export function rgbToYCbCr(r: number, g: number, b: number): [number, number, number] {
  return [
    0.299 * r + 0.587 * g + 0.114 * b,
    -0.168736 * r - 0.331264 * g + 0.5 * b,
    0.5 * r - 0.418688 * g - 0.081312 * b,
  ];
}

export function yCbCrToRgb(Y: number, Cb: number, Cr: number): [number, number, number] {
  return [Y + 1.402 * Cr, Y - 0.344136 * Cb - 0.714136 * Cr, Y + 1.772 * Cb];
}
