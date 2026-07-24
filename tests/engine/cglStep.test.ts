import { describe, it, expect } from "vitest";
import { runCGL } from "../../src/engine/cglStep";
import type { CGLParams } from "../../src/engine/params";

const L = 48;

// 決定論的な種（単位円上の位相＋滑らかな摂動＋擬似ノイズ）。
function seed(): { re: Float32Array; im: Float32Array } {
  const re = new Float32Array(L * L);
  const im = new Float32Array(L * L);
  for (let y = 0; y < L; y++) {
    for (let x = 0; x < L; x++) {
      const i = y * L + x;
      const smooth = Math.sin((2 * Math.PI * x) / L) * Math.cos((2 * Math.PI * y) / L);
      const noise = Math.sin(i * 12.9898) * 43758.5453;
      const f = 0.5 * smooth + 0.15 * (noise - Math.floor(noise));
      re[i] = Math.cos(2 * Math.PI * f);
      im[i] = Math.sin(2 * Math.PI * f);
    }
  }
  return { re, im };
}

function stats(re: Float32Array, im: Float32Array) {
  let sum = 0;
  let max = 0;
  let finite = true;
  const mags = new Float64Array(re.length);
  for (let i = 0; i < re.length; i++) {
    const a = Math.hypot(re[i], im[i]);
    if (!Number.isFinite(a)) finite = false;
    mags[i] = a;
    sum += a;
    if (a > max) max = a;
  }
  const mean = sum / re.length;
  let v = 0;
  for (const a of mags) v += (a - mean) ** 2;
  return { mean, std: Math.sqrt(v / re.length), max, finite };
}

function evolve(p: CGLParams, tEnd: number) {
  const { re, im } = seed();
  runCGL(re, im, L, p, Math.round(tEnd / p.dt));
  return stats(re, im);
}

describe("CGL 実空間陽解法（checks/cgl-explicit.py の再現）", () => {
  it("Benjamin-Feir 安定 (b=c=0.5, dt=0.02) は |ψ|→1 に飽和", () => {
    const s = evolve({ b: 0.5, c: 0.5, D: 4, dt: 0.02, speed: 1 }, 20);
    expect(s.finite).toBe(true);
    expect(s.mean).toBeGreaterThan(0.9);
    expect(s.mean).toBeLessThan(1.1);
    expect(s.std).toBeLessThan(0.1); // 一様振幅
  });

  it("乱流 (b=2, c=-1) は dt=0.01 で有界（|ψ|~1・構造あり）", () => {
    const s = evolve({ b: 2, c: -1, D: 4, dt: 0.01, speed: 1 }, 20);
    expect(s.finite).toBe(true);
    expect(s.max).toBeLessThan(2); // 有界
    expect(s.mean).toBeGreaterThan(0.3);
    expect(s.std).toBeGreaterThan(0.05); // 一様でない＝模様
  });

  it("乱流を dt=0.02 で回すと過増幅（dt≤0.01 が必要という設計判断の回帰ガード）", () => {
    const ok = evolve({ b: 2, c: -1, D: 4, dt: 0.01, speed: 1 }, 20);
    const bad = evolve({ b: 2, c: -1, D: 4, dt: 0.02, speed: 1 }, 20);
    expect(bad.max).toBeGreaterThan(ok.max * 1.3); // dt=0.02 は明確に大きく振れる
  });
});
