// 画像 → { 原本テクスチャ(線形光 RGBA), 複素種 ψ0 }。
//
// 規律（設計ノート checks/algo.py）:
//  - sRGB は線形光へ（拡散/移流は物理的な混色＝線形光で）。
//  - ダウンサンプルは面積平均（点サンプリングはモアレを作る）。
//  - 複素種は既定「色に写真」ψ0=OKLab(a,b)（色相→位相・彩度→振幅・checks/color-field.py）。

import { srgbToLinear, linearToSrgb, luminance709, linearToOklab } from "../color";
import { rgbToYCbCr } from "../engine/dynamics";

// ImageData 互換の構造型（テストでは手組みオブジェクトを渡せる＝jsdom 不要）。
export interface ImageLike {
  data: Uint8ClampedArray | Uint8Array | number[];
  width: number;
  height: number;
}

// color=写真の色をそのまま複素場に（ψ=OKLab(a,b): 色相→位相・彩度→振幅）＝既定。
//        t=0 で写真の色みが出る。明度 L は 2 自由度に載らず捨てる（表示は一定明度）。
// phase=輝度だけを位相に（|ψ|=1・色は LUT が作る・灰色写真でも構造が出る）。
// amp  =輝度だけを振幅に（位相は一様）。
export type Seed = "color" | "phase" | "amp";

// 位相種（seed=phase）の位相幅。2π にすると輝度 0 と 1 が同じ位相になり、人工的な
// 位相欠陥が生まれて自壊する（設計ノート checks/cgl_check2.py）。巻き戻しの無い幅にする。
export const PHASE_SPAN = 0.8 * Math.PI;

// 振幅種（seed=amp）の下限。|ψ|=0 は位相が未定義＝人工的な欠陥になるので避ける。
export const AMP_FLOOR = 0.15;

// color 種の彩度基準。C/C_REF を |ψ| に（純色 C≈0.12–0.31 は飽和・淡色は部分彩度）。
// 灰色(C≈0)は自然に ψ≈0＝無彩色（設計ノート checks/color-field.py）。
export const CHROMA_REF = 0.12;

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
export function imageToField(img: ImageLike, L: number, seed: Seed = "color"): Field {
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
    if (seed === "color") {
      // 写真の色をそのまま: ψ = OKLab(a,b)/C_REF。arg ψ=色相・|ψ|=彩度。
      const [, oa, ob] = linearToOklab(r, g, b);
      const c = Math.hypot(oa, ob);
      const s = c > 0 ? Math.min(1, c / CHROMA_REF) / c : 0; // |ψ|=min(1,C/C_REF)
      psiRe[i] = oa * s;
      psiIm[i] = ob * s;
    } else if (seed === "phase") {
      const y = luminance709(r, g, b); // 線形光輝度 [0,1]
      const th = PHASE_SPAN * (y - 0.5); // 巻き戻さない（±0.4π）
      psiRe[i] = Math.cos(th);
      psiIm[i] = Math.sin(th);
    } else {
      const y = luminance709(r, g, b);
      psiRe[i] = AMP_FLOOR + (1 - AMP_FLOOR) * y; // 0 を避けた振幅
      psiIm[i] = 0;
    }
  }
  return { orig, psiRe, psiIm, L };
}

// --- 力学別の種（orig=線形光 RGBA L*L*4 から状態 re/im を作る） ----------------

// 決定的ハッシュノイズ [0,1)（リセットで同じ＝再現的）。
function hashNoise(i: number): number {
  const s = Math.sin(i * 12.9898) * 43758.5453;
  return s - Math.floor(s);
}

// Gray-Scott: u=1−0.5·輝度, v=0.25·輝度＋ノイズ。ノイズで一様領域(空など)が v=0 へ
// 減衰して真っ黒になるのを防ぐ（Turing を全域で立てる・checks/gs-seed で確認）。
export function seedGrayScott(orig: Float32Array, L: number): { re: Float32Array; im: Float32Array } {
  const n = L * L;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const m = luminance709(orig[i * 4], orig[i * 4 + 1], orig[i * 4 + 2]);
    const v = Math.min(0.6, 0.22 * m + 0.14 * hashNoise(i));
    re[i] = 1 - 0.5 * v;
    im[i] = v;
  }
  return { re, im };
}

// Lenia: u=輝度 ∈ [0,1]（im は未使用）。
export function seedLenia(orig: Float32Array, L: number): { re: Float32Array; im: Float32Array } {
  const n = L * L;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    re[i] = Math.min(1, Math.max(0, luminance709(orig[i * 4], orig[i * 4 + 1], orig[i * 4 + 2])));
  }
  return { re, im };
}

// 四元数（全色発展）: q=(w,x,y,z) を interleave した Float32Array（n*4）。
// 純虚部 Im=(x,y,z) に写真の OKLab (L,a,b) を載せる（表示の逆変換）: x=2(L−0.5), y=a/0.14, z=b/0.14。
// w=0 始動（w は色に使わず内部自由度として発展）。
export function seedQuat(orig: Float32Array, L: number): Float32Array<ArrayBuffer> {
  const n = L * L;
  const q = new Float32Array(n * 4);
  const cl = (v: number): number => Math.min(1, Math.max(-1, v));
  for (let i = 0; i < n; i++) {
    const [ol, oa, ob] = linearToOklab(orig[i * 4], orig[i * 4 + 1], orig[i * 4 + 2]);
    q[i * 4] = 0; // w
    q[i * 4 + 1] = cl(2 * (ol - 0.5)); // x ← L
    q[i * 4 + 2] = cl(oa / 0.14); // y ← a（表示の彩度スケールと一致）
    q[i * 4 + 3] = cl(ob / 0.14); // z ← b
  }
  return q;
}

// 色差拡散: re=Cb, im=Cr（sRGB BT.601）。輝度は表示で原本から取るので状態に持たない。
export function seedChroma(orig: Float32Array, L: number): { re: Float32Array; im: Float32Array } {
  const n = L * L;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = linearToSrgb(orig[i * 4]);
    const g = linearToSrgb(orig[i * 4 + 1]);
    const b = linearToSrgb(orig[i * 4 + 2]);
    const [, cb, cr] = rgbToYCbCr(r, g, b);
    re[i] = cb;
    im[i] = cr;
  }
  return { re, im };
}
