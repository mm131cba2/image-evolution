// CGL コアの CPU 参照実装（Task 6 の数値の真実）。
// GPU（WGSL）はこれを 1:1 で移植する。ここで飽和・安定域を単体テストして接地する。
//
// ∂ψ/∂t = ψ + (1+ib)·D·∇²ψ − (1+ic)·|ψ|²·ψ
// 既定＝実空間ステンシル前進オイラー（近傍参照のみ・FFT 不要）。壁=反射(Neumann)ラプラシアン。
// 安定域: dt≈0.01 で乱流も有界、dt=0.02 は安定域(b小)のみ（checks/cgl-explicit.py と一致）。

import type { CGLParams } from "./params";

// 壁(Neumann=反射)境界の 5 点ラプラシアン。端は自分自身を鏡映（勾配ゼロ）。
function laplacianReflect(f: Float32Array, L: number, out: Float32Array): void {
  for (let y = 0; y < L; y++) {
    const yUp = y > 0 ? y - 1 : 0;
    const yDn = y < L - 1 ? y + 1 : L - 1;
    for (let x = 0; x < L; x++) {
      const xL = x > 0 ? x - 1 : 0;
      const xR = x < L - 1 ? x + 1 : L - 1;
      const c = y * L + x;
      out[c] = f[y * L + xL] + f[y * L + xR] + f[yUp * L + x] + f[yDn * L + x] - 4 * f[c];
    }
  }
}

// ψ を 1 ステップ前進（re/im をその場で更新）。scratch は再利用バッファ（lapRe, lapIm）。
export function cglStep(
  re: Float32Array,
  im: Float32Array,
  L: number,
  p: CGLParams,
  scratch: { lapRe: Float32Array; lapIm: Float32Array },
): void {
  const { b, c, D, dt } = p;
  const { lapRe, lapIm } = scratch;
  laplacianReflect(re, L, lapRe);
  laplacianReflect(im, L, lapIm);
  for (let i = 0; i < re.length; i++) {
    const r = re[i];
    const m = im[i];
    const m2 = r * r + m * m;
    // (1+ib)·D·∇²ψ
    const diffRe = D * (lapRe[i] - b * lapIm[i]);
    const diffIm = D * (lapIm[i] + b * lapRe[i]);
    // −(1+ic)·|ψ|²·ψ = −m2·[(r−c·m) + i(m+c·r)]
    const nlRe = -m2 * (r - c * m);
    const nlIm = -m2 * (m + c * r);
    // 成長 ψ + 拡散 + 非線形
    re[i] = r + dt * (r + diffRe + nlRe);
    im[i] = m + dt * (m + diffIm + nlIm);
  }
}

export function makeScratch(L: number): { lapRe: Float32Array; lapIm: Float32Array } {
  return { lapRe: new Float32Array(L * L), lapIm: new Float32Array(L * L) };
}

// 便宜: steps 回進める（テスト・非リアルタイム用）。
export function runCGL(re: Float32Array, im: Float32Array, L: number, p: CGLParams, steps: number): void {
  const scratch = makeScratch(L);
  for (let s = 0; s < steps; s++) cglStep(re, im, L, p, scratch);
}
