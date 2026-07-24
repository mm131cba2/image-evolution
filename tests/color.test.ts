import { describe, it, expect } from "vitest";
import {
  srgbToLinear,
  linearToSrgb,
  luminance709,
  linearToOklab,
  oklabToLinear,
  oklchToLinear,
} from "../src/color";

describe("sRGB ↔ 線形光", () => {
  it("端点は保存", () => {
    expect(srgbToLinear(0)).toBe(0);
    expect(srgbToLinear(1)).toBeCloseTo(1, 6);
    expect(linearToSrgb(0)).toBe(0);
    expect(linearToSrgb(1)).toBeCloseTo(1, 6);
  });

  it("中間グレー 0.5 は線形で ~0.214（ガンマの効き）", () => {
    expect(srgbToLinear(0.5)).toBeCloseTo(0.2140, 3);
  });

  it("往復でほぼ一致", () => {
    for (const v of [0.1, 0.25, 0.5, 0.73, 0.9]) {
      expect(linearToSrgb(srgbToLinear(v))).toBeCloseTo(v, 6);
    }
  });
});

describe("luminance709", () => {
  it("白は 1・黒は 0・緑が最大寄与", () => {
    expect(luminance709(1, 1, 1)).toBeCloseTo(1, 6);
    expect(luminance709(0, 0, 0)).toBe(0);
    expect(luminance709(0, 1, 0)).toBeGreaterThan(luminance709(1, 0, 0));
  });
});

describe("OKLab", () => {
  it("線形→OKLab→線形 で往復一致", () => {
    for (const rgb of [
      [0.2, 0.5, 0.8],
      [0.9, 0.1, 0.3],
      [0.5, 0.5, 0.5],
    ] as const) {
      const [L, a, b] = linearToOklab(rgb[0], rgb[1], rgb[2]);
      const back = oklabToLinear(L, a, b);
      for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(rgb[i], 5);
    }
  });

  it("無彩色は a=b=0", () => {
    const [, a, b] = linearToOklab(0.4, 0.4, 0.4);
    expect(a).toBeCloseTo(0, 6);
    expect(b).toBeCloseTo(0, 6);
  });

  it("OKLCh は C=0 で無彩色（色相に依らず同じ灰）", () => {
    const g0 = oklchToLinear(0.6, 0, 0);
    const g1 = oklchToLinear(0.6, 0, 2);
    for (let i = 0; i < 3; i++) expect(g0[i]).toBeCloseTo(g1[i], 6);
  });
});
