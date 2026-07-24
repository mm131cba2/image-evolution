import { describe, it, expect } from "vitest";
import { buildCyclicLUT, LUT_SIZE, CYCLIC_L, CYCLIC_C } from "../../src/render/palette";
import { srgbToLinear, linearToOklab, oklchToLinear, inGamut } from "../../src/color";

function labOf(lut: Uint8Array, i: number): [number, number, number] {
  const r = srgbToLinear(lut[i * 4] / 255);
  const g = srgbToLinear(lut[i * 4 + 1] / 255);
  const b = srgbToLinear(lut[i * 4 + 2] / 255);
  return linearToOklab(r, g, b);
}

function deltaE(a: [number, number, number], b: [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

describe("buildCyclicLUT", () => {
  const lut = buildCyclicLUT();

  it("256×4 RGBA・アルファ 255", () => {
    expect(lut.length).toBe(LUT_SIZE * 4);
    for (let i = 0; i < LUT_SIZE; i++) expect(lut[i * 4 + 3]).toBe(255);
  });

  it("全色相が sRGB 域内（clip で色相が飛ばない）", () => {
    for (let i = 0; i < LUT_SIZE; i++) {
      const h = (2 * Math.PI * i) / LUT_SIZE;
      expect(inGamut(oklchToLinear(CYCLIC_L, CYCLIC_C, h))).toBe(true);
    }
  });

  it("連続 OKLCh の知覚ステップ ΔE がほぼ一定（設計の均等性・CV < 0.05）", () => {
    // 実際の均等性はマッピング（量子化前）の性質。バイト LUT の CV は微小ステップに
    // 対する丸め雑音が支配して非本質なので、連続値で測る（HSV は 0.51・checks/complex-color.py）。
    const steps: number[] = [];
    for (let i = 0; i < LUT_SIZE; i++) {
      const h0 = (2 * Math.PI * i) / LUT_SIZE;
      const h1 = (2 * Math.PI * ((i + 1) % LUT_SIZE)) / LUT_SIZE;
      const l0 = oklchToLinear(CYCLIC_L, CYCLIC_C, h0);
      const l1 = oklchToLinear(CYCLIC_L, CYCLIC_C, h1);
      steps.push(deltaE(linearToOklab(l0[0], l0[1], l0[2]), linearToOklab(l1[0], l1[1], l1[2])));
    }
    const mean = steps.reduce((a, b) => a + b, 0) / steps.length;
    const varc = steps.reduce((a, b) => a + (b - mean) ** 2, 0) / steps.length;
    expect(mean).toBeGreaterThan(0);
    expect(Math.sqrt(varc) / mean).toBeLessThan(0.05);
  });

  it("継ぎ目（255→0）も他と同程度＝連続に一周", () => {
    const seam = deltaE(labOf(lut, LUT_SIZE - 1), labOf(lut, 0));
    const typical = deltaE(labOf(lut, 10), labOf(lut, 11));
    expect(seam).toBeLessThan(typical * 3);
  });
});
