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

// morph λ∈[0,1]＝diffusion↔Lenia 核 morph（統合形の橋・checks/kernel-morph.py）:
//   λ=1（既定）… リング核(kr0=0.5)＋ベル成長＝Lenia の生命的パターン。
//   λ=0        … 中心核(kr0=0＝近距離平均)＋恒等成長＝局所平均への緩和＝拡散(均す・§3 K∗u−u→∇²)。
//   中間        … λ_c≈0.6 でパターンが分岐（Turing 型・§5）。核と成長を同時に morph。
export function leniaStep(re: Float32Array, im: Float32Array, L: number, morph = 1): void {
  const n = L * L;
  const out = new Float32Array(n);
  const { mu, sigma, R, kr0, kw, dt } = LENIA;
  const decay = Math.exp(-dt);
  const kr = kr0 * morph; // リング半径を λ で 0（中心＝局所）↔0.5（リング）に morph
  for (let y = 0; y < L; y++) {
    for (let x = 0; x < L; x++) {
      let accWU = 0;
      let accW = 0;
      for (let dy = -R; dy <= R; dy++) {
        const yy = Math.min(L - 1, Math.max(0, y + dy));
        for (let dx = -R; dx <= R; dx++) {
          const r = Math.hypot(dx, dy) / R;
          if (r > 1 || r === 0) continue;
          const w = bell(r, kr, kw);
          const xx = Math.min(L - 1, Math.max(0, x + dx));
          accWU += w * re[yy * L + xx];
          accW += w;
        }
      }
      const pot = accW > 0 ? accWU / accW : 0; // 正規化ポテンシャル ∈ [0,1]
      // 成長目標を 恒等(pot＝拡散) ↔ ベル(Lenia) で morph。
      const g = (1 - morph) * pot + morph * bell(pot, mu, sigma);
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
//
// coupling λ∈[0,1] が「代数結合の強さ」＝scalar↔quat の連続ノブ（checks/color-algebra.py）:
//   λ=1（既定）… 完全な四元数 CGL。回転 (1+bI)/(1+cI) と共有ノルム |q|² が全成分を常時混ぜる
//                 ＝色相回転が内蔵・彩度を保つ。
//   λ=0        … 回転を切り(b,c→0)ノルムを成分ごと(q_k²)に＝4 本の独立な実 Stuart–Landau 場の
//                 直和 ℝ⊕ℝ⊕ℝ⊕ℝ＝成分が混ざらない・独立拡散で褪色。
//   中間        … 結合が無段階に立つ（成分漏れ 0↔0.49・color-algebra.py 実測）。
export function quatCglStep(
  q: Float32Array,
  L: number,
  p: { b: number; c: number; D: number; dt: number; coupling?: number },
): void {
  const n = L * L;
  const { b, c, D, dt } = p;
  const lam = p.coupling ?? 1; // 既定は完全結合（従来挙動を保つ）
  const bb = lam * b, cc = lam * c; // 回転係数を λ で絞る（λ=0 で恒等＝混ぜない）
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
    // ノルムを 共有|q|² ↔ 成分ごとq_k² で補間（λ=0 で各成分が自分の二乗＝独立飽和）。
    const nw2 = (1 - lam) * w * w + lam * m2;
    const nx2 = (1 - lam) * x * x + lam * m2;
    const ny2 = (1 - lam) * y * y + lam * m2;
    const nz2 = (1 - lam) * z * z + lam * m2;
    const lw = lap[0][i], lx = lap[1][i], ly = lap[2][i], lz = lap[3][i];
    // (1+λbI)⊗(D·∇²q)  … λ=0 で恒等＝成分ごとの拡散（混ざらない）
    const [dw, dx, dy, dz] = qmul(1, bb * Ix, bb * Iy, bb * Iz, D * lw, D * lx, D * ly, D * lz);
    // (1+λcI)⊗(n²⊙q)   … λ=0 で恒等かつ n²=q_k²＝成分ごとの立方飽和
    const [nw, nx, ny, nz] = qmul(1, cc * Ix, cc * Iy, cc * Iz, nw2 * w, nx2 * x, ny2 * y, nz2 * z);
    q[i * 4] = w + dt * (w + dw - nw);
    q[i * 4 + 1] = x + dt * (x + dx - nx);
    q[i * 4 + 2] = y + dt * (y + dy - ny);
    q[i * 4 + 3] = z + dt * (z + dz - nz);
  }
}

// ---------------------------------------------------------------------------
// 統合形（四元数版）。状態 q=(w,x,y,z)。統合形 ∂q=α0·q+(1+λbI)(αL∇²q)+α4∇⁴q−(1+λcI)(n²⊙q) を
// ℍ 上で 1 ステップ実装。3 直交ノブで統合形の 3 軸を走査（checks/unified-quat.py・unified-telegraph.py）:
//   coupling λ … 場の代数（scalar ℝ⁴ ↔ quat ℍ・quatCglStep と同じ）。
//   pattern p  … σ(k) ピーク位置（CGL 拡散 k=0 リミットサイクル ↔ SH −(1+∇²)² 有限 k）。
//                係数 morph（mix(a,b,p)=a+(b−a)p・r=0.5）: α0=mix(1,r−1,p)・αL=mix(D,−2,p)・α4=−p。
//   wave a     … 時間の階数（拡散/緩和 ↔ 波動）。移流(拡散項 T)だけに 1 極慣性フィルタ
//                v←a·v+(1−a)·T を掛け、反応(自己増殖+∇⁴+cubic)は 1 階に保つ＝双曲型反応拡散
//                （Cattaneo・飽和で有界）。vel を渡し a>0 のとき有効。
//   すべて「off」端（p=0, a=0）で quatCglStep と厳密一致（strict superset）。
//   dt は p が立つほど 0.02 で頭打ち（SH の陽的安定・GPU と一致）。
// ---------------------------------------------------------------------------
export function unifiedQuatStep(
  q: Float32Array,
  L: number,
  p: { b: number; c: number; D: number; dt: number; coupling?: number; pattern?: number; wave?: number; vel?: Float32Array },
): void {
  const n = L * L;
  const { b, c, D } = p;
  const lam = p.coupling ?? 1;
  const pat = p.pattern ?? 0;
  const a = p.wave ?? 0;
  const vel = p.vel;
  const r = 0.5;
  const a0 = 1 + (r - 2) * pat;      // mix(1, r−1, pat)＝1−1.5p（p=1 で SH の自己項 r−1=−0.5）
  const aL = D + (-2 - D) * pat;      // mix(D, −2, pat)（p=1 で SH の −2∇²）
  const a4 = -pat;                    // mix(0, −1, pat)（p=1 で SH の −∇⁴）
  const dt = p.dt + (Math.min(p.dt, 0.02) - p.dt) * pat; // mix(dt, min(dt,0.02), pat)
  const bb = lam * b, cc = lam * c;
  const [Ix, Iy, Iz] = QUAT_AXIS;
  const comp = new Float32Array(n);
  const lap = [new Float32Array(n), new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  const bih = [new Float32Array(n), new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  for (let k = 0; k < 4; k++) {
    for (let i = 0; i < n; i++) comp[i] = q[i * 4 + k];
    lapReflect(comp, L, lap[k]);
    biharmReflect(comp, L, bih[k]);
  }
  const useWave = a > 0 && vel !== undefined;
  for (let i = 0; i < n; i++) {
    const w = q[i * 4], x = q[i * 4 + 1], y = q[i * 4 + 2], z = q[i * 4 + 3];
    const m2 = w * w + x * x + y * y + z * z;
    const nw2 = (1 - lam) * w * w + lam * m2;
    const nx2 = (1 - lam) * x * x + lam * m2;
    const ny2 = (1 - lam) * y * y + lam * m2;
    const nz2 = (1 - lam) * z * z + lam * m2;
    const lw = lap[0][i], lx = lap[1][i], ly = lap[2][i], lz = lap[3][i];
    // 移流 T=(1+λbI)⊗(αL∇²q)・反応 Rx=α0·q+α4∇⁴q−(1+λcI)(n²⊙q)。
    const [dw, dx, dy, dz] = qmul(1, bb * Ix, bb * Iy, bb * Iz, aL * lw, aL * lx, aL * ly, aL * lz);
    const [nw, nx, ny, nz] = qmul(1, cc * Ix, cc * Iy, cc * Iz, nw2 * w, nx2 * x, ny2 * y, nz2 * z);
    const Rw = a0 * w + a4 * bih[0][i] - nw;
    const Rx = a0 * x + a4 * bih[1][i] - nx;
    const Ry = a0 * y + a4 * bih[2][i] - ny;
    const Rz = a0 * z + a4 * bih[3][i] - nz;
    if (useWave) {
      const vw = a * vel![i * 4] + (1 - a) * dw; vel![i * 4] = vw;
      const vx = a * vel![i * 4 + 1] + (1 - a) * dx; vel![i * 4 + 1] = vx;
      const vy = a * vel![i * 4 + 2] + (1 - a) * dy; vel![i * 4 + 2] = vy;
      const vz = a * vel![i * 4 + 3] + (1 - a) * dz; vel![i * 4 + 3] = vz;
      q[i * 4] = w + dt * (vw + Rw);
      q[i * 4 + 1] = x + dt * (vx + Rx);
      q[i * 4 + 2] = y + dt * (vy + Ry);
      q[i * 4 + 3] = z + dt * (vz + Rz);
    } else {
      q[i * 4] = w + dt * (dw + Rw);
      q[i * 4 + 1] = x + dt * (dx + Rx);
      q[i * 4 + 2] = y + dt * (dy + Ry);
      q[i * 4 + 3] = z + dt * (dz + Rz);
    }
  }
}

// 壁(clamp)境界の 13 点 biharmonic ∇⁴u = ∇²(∇²u)（5点ラプラシアンの自己合成）。
// = 20c −8(N+S+E+W) +2(斜め4) +(NN+SS+EE+WW)。SH/CH の 4 階項に使う。
function biharmReflect(f: Float32Array, L: number, out: Float32Array): void {
  const cl = (v: number): number => (v < 0 ? 0 : v > L - 1 ? L - 1 : v);
  const at = (x: number, y: number): number => f[cl(y) * L + cl(x)];
  for (let y = 0; y < L; y++) {
    for (let x = 0; x < L; x++) {
      out[y * L + x] =
        20 * f[y * L + x] -
        8 * (at(x, y - 1) + at(x, y + 1) + at(x - 1, y) + at(x + 1, y)) +
        2 * (at(x + 1, y - 1) + at(x - 1, y - 1) + at(x + 1, y + 1) + at(x - 1, y + 1)) +
        (at(x, y - 2) + at(x, y + 2) + at(x - 2, y) + at(x + 2, y));
    }
  }
}

// ---------------------------------------------------------------------------
// 波動方程式（減衰つき）。状態 re=u（変位）, im=v（速度）。写真=初期変位で波紋が伝播。
// u_t = v,  v_t = c²∇²u − γv。checks 済み: 有界に鳴って徐々に静まる。
// ---------------------------------------------------------------------------
export const WAVE = { c2: 0.2, g: 0.004, dt: 0.2 } as const;
export function waveStep(re: Float32Array, im: Float32Array, L: number): void {
  const n = L * L;
  const lu = new Float32Array(n);
  lapReflect(re, L, lu);
  const { c2, g, dt } = WAVE;
  for (let i = 0; i < n; i++) {
    im[i] = im[i] + dt * (c2 * lu[i] - g * im[i]);
    re[i] = re[i] + dt * im[i];
  }
}

// ---------------------------------------------------------------------------
// Swift-Hohenberg（縞・迷路・六方）。状態 re=u（im 未使用）。選択波長でパターン形成。
// u_t = (r − (1+∇²)²)u − u³ = r·u − u − 2∇²u − ∇⁴u − u³。checks 済み: ±1 縞に自己組織化。
// ---------------------------------------------------------------------------
export const SH = { r: 0.5, dt: 0.02 } as const;
export function swiftHohenbergStep(re: Float32Array, im: Float32Array, L: number): void {
  const n = L * L;
  const lu = new Float32Array(n);
  const b4 = new Float32Array(n);
  lapReflect(re, L, lu);
  biharmReflect(re, L, b4);
  const { r, dt } = SH;
  for (let i = 0; i < n; i++) {
    const u = re[i];
    re[i] = u + dt * (r * u - u - 2 * lu[i] - b4[i] - u * u * u);
  }
  im.fill(0);
}

// ---------------------------------------------------------------------------
// FitzHugh-Nagumo（興奮性・伝播波）。状態 re=u, im=v。振動域で伝播する波。
// u_t = u − u³/3 − v + D∇²u,  v_t = ε(u + a − b·v)。checks 済み: 伝播波（のち同期）。
// ---------------------------------------------------------------------------
export const FHN = { D: 1, eps: 0.08, a: 0.2, b: 0.5, dt: 0.12 } as const;
export function fitzHughNagumoStep(re: Float32Array, im: Float32Array, L: number): void {
  const n = L * L;
  const lu = new Float32Array(n);
  lapReflect(re, L, lu);
  const { D, eps, a, b, dt } = FHN;
  for (let i = 0; i < n; i++) {
    const u = re[i];
    const v = im[i];
    re[i] = u + dt * (u - (u * u * u) / 3 - v + D * lu[i]);
    im[i] = v + dt * eps * (u + a - b * v);
  }
}

// ---------------------------------------------------------------------------
// Cahn-Hilliard（相分離）。状態 re=u（im 未使用）。写真が ±1 ドメインに分離し粗大化。
// u_t = ∇²(u³ − u − κ∇²u) = ∇²(u³) − ∇²u − κ∇⁴u（質量保存）。checks 済み: ±1 に相分離。
// ---------------------------------------------------------------------------
export const CH = { k: 0.5, dt: 0.008 } as const;
export function cahnHilliardStep(re: Float32Array, im: Float32Array, L: number): void {
  const n = L * L;
  const u3 = new Float32Array(n);
  for (let i = 0; i < n; i++) u3[i] = re[i] * re[i] * re[i];
  const lu3 = new Float32Array(n);
  const lu = new Float32Array(n);
  const b4 = new Float32Array(n);
  lapReflect(u3, L, lu3);
  lapReflect(re, L, lu);
  biharmReflect(re, L, b4);
  const { k, dt } = CH;
  for (let i = 0; i < n; i++) {
    re[i] = re[i] + dt * (lu3[i] - lu[i] - k * b4[i]);
  }
  im.fill(0);
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
