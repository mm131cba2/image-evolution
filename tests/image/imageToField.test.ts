import { describe, it, expect } from "vitest";
import { imageToField, resampleBox, type ImageLike } from "../../src/image/imageToField";
import { srgbToLinear } from "../../src/color";

// RGBA バイト画像を手組み（jsdom 不要）。fn(x,y)->[r,g,b] は 0..255。
function makeImage(w: number, h: number, fn: (x: number, y: number) => [number, number, number]): ImageLike {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b] = fn(x, y);
      const i = (y * w + x) * 4;
      data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
    }
  }
  return { data, width: w, height: h };
}

describe("resampleBox（面積平均）", () => {
  it("市松 2×2 を 1×1 に縮小すると平均 0.5（点サンプルの 0/1 でない）", () => {
    // 白黒市松。面積平均なら 0.5、点サンプルなら 0 か 1 に張り付く＝モアレ源。
    const src = new Float32Array([1, 0, 0, 1]); // 1ch 2x2
    const out = resampleBox(src, 2, 2, 1, 1, 1);
    expect(out[0]).toBeCloseTo(0.5, 6);
  });

  it("一様画像はサイズを変えても値を保つ", () => {
    const src = new Float32Array(16 * 16).fill(0.37);
    const out = resampleBox(src, 16, 16, 5, 5, 1);
    for (const v of out) expect(v).toBeCloseTo(0.37, 6);
  });

  it("ブロック平均を保つ（左半分1・右半分0 の 4×1 → 2×1）", () => {
    const src = new Float32Array([1, 1, 0, 0]);
    const out = resampleBox(src, 4, 1, 2, 1, 1);
    expect(out[0]).toBeCloseTo(1, 6);
    expect(out[1]).toBeCloseTo(0, 6);
  });
});

describe("imageToField", () => {
  it("出力サイズが L×L", () => {
    const img = makeImage(8, 8, () => [128, 128, 128]);
    const f = imageToField(img, 4);
    expect(f.L).toBe(4);
    expect(f.orig.length).toBe(4 * 4 * 4);
    expect(f.psiRe.length).toBe(16);
    expect(f.psiIm.length).toBe(16);
  });

  it("原本は線形光に変換される（sRGB 128 → 線形 ~0.216）", () => {
    const img = makeImage(4, 4, () => [128, 128, 128]);
    const f = imageToField(img, 2);
    const expected = srgbToLinear(128 / 255);
    expect(f.orig[0]).toBeCloseTo(expected, 4);
    expect(f.orig[3]).toBe(1); // アルファ
  });

  it("位相種は |ψ|=1（単位円上）", () => {
    const img = makeImage(4, 4, (x) => [x * 60, 100, 200]);
    const f = imageToField(img, 4, "phase");
    for (let i = 0; i < f.psiRe.length; i++) {
      const mag = Math.hypot(f.psiRe[i], f.psiIm[i]);
      expect(mag).toBeCloseTo(1, 6);
    }
  });

  it("振幅種は虚部ゼロ・実部が輝度", () => {
    const img = makeImage(4, 4, () => [255, 255, 255]);
    const f = imageToField(img, 2, "amp");
    for (let i = 0; i < f.psiIm.length; i++) expect(f.psiIm[i]).toBe(0);
    // 白＝線形 RGB 1 → 輝度 1
    expect(f.psiRe[0]).toBeCloseTo(1, 5);
  });
});
