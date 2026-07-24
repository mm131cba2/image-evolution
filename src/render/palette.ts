// モード B（複素場 ψ の表示）の知覚均等な循環カラーマップ。
// 位相 arg(ψ) → 色相角、振幅 |ψ| → 彩度/明度（表示シェーダー側で変調）。
// HSV は色相の知覚ステップが 14.2 倍ばらつくので不可。OKLCh で ΔE 一定に（checks/complex-color.py）。

import { oklchToLinear, linearToSrgb } from "../color";

export const LUT_SIZE = 256;

// 全色相で sRGB 域内に収まる固定 L,C。checks/complex-color.py: L=0.72 の最大彩度は
// 色相全域で最小 0.123 なので C=0.11 は安全（clip なし＝色相が飛ばない）。
export const CYCLIC_L = 0.72;
export const CYCLIC_C = 0.11;

// 位相 [0,2π) → sRGB の 256 段循環 LUT（RGBA バイト）。
export function buildCyclicLUT(): Uint8Array<ArrayBuffer> {
  const lut = new Uint8Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    const h = (2 * Math.PI * i) / LUT_SIZE;
    const lin = oklchToLinear(CYCLIC_L, CYCLIC_C, h);
    lut[i * 4] = Math.round(linearToSrgb(lin[0]) * 255);
    lut[i * 4 + 1] = Math.round(linearToSrgb(lin[1]) * 255);
    lut[i * 4 + 2] = Math.round(linearToSrgb(lin[2]) * 255);
    lut[i * 4 + 3] = 255;
  }
  return lut;
}
