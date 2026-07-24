// 画像 → { 原本テクスチャ(線形光 RGBA), 複素種 ψ0 }。
//
// 規律（設計ノート checks/algo.py）:
//  - sRGB は線形光へ（拡散/移流は物理的な混色＝線形光で）。
//  - ダウンサンプルは面積平均（点サンプリングはモアレを作る）。
//  - 複素種は既定「位相に写真」ψ0=exp(2πi·Y)（位相の方が CGL で記憶が残る・checks/cgl.py）。

import { srgbToLinear, luminance709 } from "../color";

// ImageData 互換の構造型（テストでは手組みオブジェクトを渡せる＝jsdom 不要）。
export interface ImageLike {
  data: Uint8ClampedArray | Uint8Array | number[];
  width: number;
  height: number;
}

export type Seed = "phase" | "amp";

// 位相種の位相幅。2π にすると輝度 0 と 1 が同じ位相になり、人工的な位相欠陥が
// 生まれて自壊する（設計ノート checks/cgl_check2.py で確認）。巻き戻しの無い幅にする。
export const PHASE_SPAN = 0.8 * Math.PI;

export interface Field {
  orig: Float32Array; // L*L*4 線形光 RGBA（A=1）。モード A のサンプル元。
  psiRe: Float32Array; // L*L
  psiIm: Float32Array; // L*L
  L: number;
}

// 面積平均リサンプル（端は座標クランプ・小数被覆を重みに）。ch チャンネル。
// 縮小は箱平均（モアレを抑える）、拡大は被覆重みで補間的に振る舞う。
export function resampleBox(
  src: Float32Array,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
  ch: number,
): Float32Array {
  const out = new Float32Array(dw * dh * ch);
  const sx = sw / dw;
  const sy = sh / dh;
  const acc = new Float64Array(ch);
  for (let dy = 0; dy < dh; dy++) {
    const y0 = dy * sy;
    const y1 = y0 + sy;
    const iy0 = Math.floor(y0);
    const iy1 = Math.ceil(y1);
    for (let dx = 0; dx < dw; dx++) {
      const x0 = dx * sx;
      const x1 = x0 + sx;
      const ix0 = Math.floor(x0);
      const ix1 = Math.ceil(x1);
      acc.fill(0);
      let area = 0;
      for (let iy = iy0; iy < iy1; iy++) {
        const wy = Math.min(iy + 1, y1) - Math.max(iy, y0);
        if (wy <= 0) continue;
        const cy = Math.min(Math.max(iy, 0), sh - 1);
        for (let ix = ix0; ix < ix1; ix++) {
          const wx = Math.min(ix + 1, x1) - Math.max(ix, x0);
          if (wx <= 0) continue;
          const cx = Math.min(Math.max(ix, 0), sw - 1);
          const w = wx * wy;
          const si = (cy * sw + cx) * ch;
          for (let c = 0; c < ch; c++) acc[c] += src[si + c] * w;
          area += w;
        }
      }
      const di = (dy * dw + dx) * ch;
      const inv = area > 0 ? 1 / area : 0;
      for (let c = 0; c < ch; c++) out[di + c] = acc[c] * inv;
    }
  }
  return out;
}

// 画像を L×L の場に変換。
export function imageToField(img: ImageLike, L: number, seed: Seed = "phase"): Field {
  const { data, width: sw, height: sh } = img;
  // sRGB バイト → 線形光 RGB（3ch）。
  const linSrc = new Float32Array(sw * sh * 3);
  for (let i = 0, p = 0; i < sw * sh; i++) {
    const b = i * 4;
    linSrc[p++] = srgbToLinear(data[b] / 255);
    linSrc[p++] = srgbToLinear(data[b + 1] / 255);
    linSrc[p++] = srgbToLinear(data[b + 2] / 255);
  }
  // 面積平均で L×L へ。
  const lin = resampleBox(linSrc, sw, sh, L, L, 3);

  const orig = new Float32Array(L * L * 4);
  const psiRe = new Float32Array(L * L);
  const psiIm = new Float32Array(L * L);
  for (let i = 0; i < L * L; i++) {
    const r = lin[i * 3];
    const g = lin[i * 3 + 1];
    const b = lin[i * 3 + 2];
    orig[i * 4] = r;
    orig[i * 4 + 1] = g;
    orig[i * 4 + 2] = b;
    orig[i * 4 + 3] = 1;
    const y = luminance709(r, g, b); // 線形光輝度 [0,1]
    if (seed === "phase") {
      const th = PHASE_SPAN * (y - 0.5); // 巻き戻さない（±0.4π）
      psiRe[i] = Math.cos(th);
      psiIm[i] = Math.sin(th);
    } else {
      psiRe[i] = y;
      psiIm[i] = 0;
    }
  }
  return { orig, psiRe, psiIm, L };
}
