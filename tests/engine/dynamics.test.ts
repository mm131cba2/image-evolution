import { describe, it, expect } from "vitest";
import {
  grayScottStep,
  leniaStep,
  chromaStep,
  rgbToYCbCr,
  yCbCrToRgb,
} from "../../src/engine/dynamics";

const maxAbs = (a: Float32Array): number => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const mean = (a: Float32Array): number => a.reduce((s, v) => s + v, 0) / a.length;
const variance = (a: Float32Array): number => {
  const m = mean(a);
  return a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length;
};

describe("grayScottStep", () => {
  it("一様な平衡状態(u=1,v=0)は動かない", () => {
    const L = 8;
    const u = new Float32Array(L * L).fill(1);
    const v = new Float32Array(L * L).fill(0);
    for (let s = 0; s < 20; s++) grayScottStep(u, v, L);
    expect(maxAbs(new Float32Array(u.map((x) => x - 1)))).toBeLessThan(1e-5);
    expect(maxAbs(v)).toBeLessThan(1e-5);
  });

  it("種を撒くと v が成長し値域は有界のまま", () => {
    const L = 16;
    const u = new Float32Array(L * L).fill(1);
    const v = new Float32Array(L * L).fill(0);
    // 中央に種
    for (let y = 6; y < 10; y++) for (let x = 6; x < 10; x++) { u[y * L + x] = 0.5; v[y * L + x] = 0.25; }
    for (let s = 0; s < 200; s++) grayScottStep(u, v, L);
    expect(mean(v)).toBeGreaterThan(0); // パターンが広がった
    // 発散しない（u,v は概ね [0,1] に留まる）
    expect(maxAbs(u)).toBeLessThan(1.2);
    expect(maxAbs(v)).toBeLessThan(1.2);
  });
});

describe("leniaStep", () => {
  it("値域 [0,1] を保ち、ゼロ場はゼロのまま（成長目標≈0）", () => {
    const L = 24;
    const u = new Float32Array(L * L).fill(0);
    const im = new Float32Array(L * L);
    for (let s = 0; s < 10; s++) leniaStep(u, im, L);
    expect(maxAbs(u)).toBeLessThan(1e-3);
  });

  it("塊を置くと動くが値域は [0,1] を出ない", () => {
    const L = 32;
    const u = new Float32Array(L * L).fill(0);
    const im = new Float32Array(L * L);
    for (let y = 12; y < 20; y++) for (let x = 12; x < 20; x++) u[y * L + x] = 1;
    const before = mean(u);
    for (let s = 0; s < 15; s++) leniaStep(u, im, L);
    for (const val of u) { expect(val).toBeGreaterThanOrEqual(-1e-4); expect(val).toBeLessThanOrEqual(1 + 1e-4); }
    expect(mean(u)).not.toBeCloseTo(before, 5); // 変化した
    expect(maxAbs(im)).toBe(0); // im は未使用
  });
});

describe("chromaStep", () => {
  it("拡散は平均を保存し分散を減らす（色差が滲む）", () => {
    const L = 16;
    const cb = new Float32Array(L * L);
    const cr = new Float32Array(L * L);
    for (let i = 0; i < L * L; i++) { cb[i] = (i % 2) * 0.4 - 0.2; cr[i] = 0.1; }
    const m0 = mean(cb), var0 = variance(cb);
    for (let s = 0; s < 30; s++) chromaStep(cb, cr, L, 4, 0.02);
    expect(mean(cb)).toBeCloseTo(m0, 4); // 保存
    expect(variance(cb)).toBeLessThan(var0); // 均された
  });
});

describe("YCbCr 往復", () => {
  it("rgb→ycbcr→rgb が恒等", () => {
    for (const c of [[0.2, 0.7, 0.4], [1, 0, 0], [0.5, 0.5, 0.5]]) {
      const [Y, Cb, Cr] = rgbToYCbCr(c[0], c[1], c[2]);
      const [r, g, b] = yCbCrToRgb(Y, Cb, Cr);
      expect(r).toBeCloseTo(c[0], 5);
      expect(g).toBeCloseTo(c[1], 5);
      expect(b).toBeCloseTo(c[2], 5);
    }
  });
});
