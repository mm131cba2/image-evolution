import { describe, it, expect } from "vitest";
import {
  downsampleField,
  l2dist,
  bestLoop,
  crossfadeWeights,
} from "../../src/engine/loop";

// P 個の相異なる特徴を周期 P で繰り返す列を作る（決定的）。
function periodicFeatures(P: number, reps: number, dim = 4): Float32Array[] {
  const base: Float32Array[] = [];
  for (let p = 0; p < P; p++) {
    const v = new Float32Array(dim);
    for (let k = 0; k < dim; k++) v[k] = Math.sin(p * 1.7 + k * 0.9);
    base.push(v);
  }
  const seq: Float32Array[] = [];
  for (let t = 0; t < P * reps; t++) seq.push(base[t % P]);
  return seq;
}

describe("l2dist", () => {
  it("同一ベクトルは 0", () => {
    const a = new Float32Array([1, -2, 3]);
    expect(l2dist(a, a)).toBe(0);
  });
  it("直交差はユークリッド距離", () => {
    expect(l2dist(new Float32Array([3, 0]), new Float32Array([0, 4]))).toBeCloseTo(5);
  });
});

describe("downsampleField", () => {
  it("特徴長は comps*K*K", () => {
    const L = 8;
    const f = new Float32Array(L * L * 2);
    expect(downsampleField(f, L, 2, 4).length).toBe(2 * 4 * 4);
  });

  it("一様場はブロック平均も一様", () => {
    const L = 8;
    const f = new Float32Array(L * L * 2).fill(0.5);
    const feat = downsampleField(f, L, 2, 2);
    for (const v of feat) expect(v).toBeCloseTo(0.5);
  });

  it("左上に偏った場は左上ブロックの値が大きい", () => {
    const L = 8;
    const comps = 1;
    const f = new Float32Array(L * L * comps);
    for (let y = 0; y < L; y++)
      for (let x = 0; x < L; x++) if (x < L / 2 && y < L / 2) f[y * L + x] = 1;
    // K=2 → ブロック配置 [TL TR; BL BR]
    const feat = downsampleField(f, L, comps, 2);
    const [tl, tr, bl, br] = feat;
    expect(tl).toBeCloseTo(1); // 左上ブロックは全 1
    expect(tr).toBeCloseTo(0);
    expect(bl).toBeCloseTo(0);
    expect(br).toBeCloseTo(0);
  });
});

describe("bestLoop", () => {
  it("厳密に周期的な列では周期を継ぎ目コスト 0 で見つける", () => {
    const P = 7;
    const seq = periodicFeatures(P, 5);
    const r = bestLoop(seq, { minLength: 2, maxLength: 3 * P, window: 2 });
    expect(r.length).toBe(P);
    expect(r.cost).toBeCloseTo(0, 5);
  });

  it("同コストなら最短周期を選ぶ（1 周期を 2 周期と誤検出しない）", () => {
    const P = 5;
    const seq = periodicFeatures(P, 6); // 2P も継ぎ目 0 だが P を返すべき
    const r = bestLoop(seq, { minLength: 2, maxLength: 4 * P, window: 1 });
    expect(r.length).toBe(P);
  });

  it("minLength より短いループは返さない", () => {
    const P = 3;
    const seq = periodicFeatures(P, 8);
    const r = bestLoop(seq, { minLength: 4, maxLength: 20, window: 1 });
    expect(r.length).toBeGreaterThanOrEqual(4);
    // P=3 の倍数（6）が最短の許容周期
    expect(r.length % P).toBe(0);
  });

  it("振幅ドリフトがあると継ぎ目コストは正だが周期は当てる", () => {
    const P = 6;
    const base = periodicFeatures(P, 6);
    // 時間とともに徐々に縮む（近似ループ）
    const seq = base.map((v, t) => {
      const g = 1 - t * 0.004;
      const out = new Float32Array(v.length);
      for (let k = 0; k < v.length; k++) out[k] = v[k] * g;
      return out;
    });
    const r = bestLoop(seq, { minLength: 2, maxLength: 3 * P, window: 2 });
    expect(r.length).toBe(P);
    expect(r.cost).toBeGreaterThan(0);
    // 周期一致（lag=P）は非周期の lag=P+1 より継ぎ目が小さい
    const at = (lag: number) => l2dist(seq[0], seq[lag]);
    expect(at(P)).toBeLessThan(at(P + 1));
  });
});

describe("crossfadeWeights", () => {
  it("長さと単調増加（0<w<1）", () => {
    const w = crossfadeWeights(4);
    expect(w.length).toBe(4);
    for (let i = 0; i < w.length; i++) {
      expect(w[i]).toBeGreaterThan(0);
      expect(w[i]).toBeLessThan(1);
      if (i > 0) expect(w[i]).toBeGreaterThan(w[i - 1]);
    }
  });
  it("fade=0 は空", () => {
    expect(crossfadeWeights(0).length).toBe(0);
  });
});
