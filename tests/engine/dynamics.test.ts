import { describe, it, expect } from "vitest";
import {
  grayScottStep,
  leniaStep,
  chromaStep,
  quatCglStep,
  waveStep,
  swiftHohenbergStep,
  fitzHughNagumoStep,
  cahnHilliardStep,
  rgbToYCbCr,
  yCbCrToRgb,
} from "../../src/engine/dynamics";

const maxAbs = (a: Float32Array): number => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const mean = (a: Float32Array): number => a.reduce((s, v) => s + v, 0) / a.length;
const variance = (a: Float32Array): number => {
  const m = mean(a);
  return a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length;
};

// 実数スカラー（全色・独立）モード＝quatCglStep の coupling=0 極限。
// 各色成分が独立な実 Stuart-Landau 場（ℝ⊕ℝ⊕ℝ⊕ℝ）＝混ざらないことを確定する
// （color-algebra.py の scalar スキーム／quat 結合ノブ λ=0 の意味づけ）。
describe("scalar モード（quatCglStep coupling=0＝実数スカラー全色）", () => {
  const build = (L: number): Float32Array => {
    const q = new Float32Array(L * L * 4); // 全成分 0 から x にだけインパルス
    q[(Math.floor(L / 2) * L + Math.floor(L / 2)) * 4 + 1] = 0.5; // x 成分
    return q;
  };
  it("λ=0 は成分が混ざらない（x のインパルスが w/y/z へ漏れない）", () => {
    const L = 16;
    const q = build(L);
    for (let s = 0; s < 20; s++) quatCglStep(q, L, { b: 0.5, c: 0.5, D: 1, dt: 0.05, coupling: 0 });
    let leak = 0;
    for (let i = 0; i < L * L; i++)
      leak = Math.max(leak, Math.abs(q[i * 4]), Math.abs(q[i * 4 + 2]), Math.abs(q[i * 4 + 3]));
    expect(leak).toBeLessThan(1e-6); // w,y,z は 0 のまま＝独立＝実数スカラー直和
  });
  it("λ=1（四元数）は結合して他成分へ漏れる（対照）", () => {
    const L = 16;
    const q = build(L);
    for (let s = 0; s < 20; s++) quatCglStep(q, L, { b: 0.5, c: 0.5, D: 1, dt: 0.05, coupling: 1 });
    let leak = 0;
    for (let i = 0; i < L * L; i++)
      leak = Math.max(leak, Math.abs(q[i * 4]), Math.abs(q[i * 4 + 2]), Math.abs(q[i * 4 + 3]));
    expect(leak).toBeGreaterThan(1e-3); // 回転で他成分へ漏れる
  });
});

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

  it("morph λ: λ=0 は拡散（細部を均す）・λ=1 は Lenia（構造を保つ）", () => {
    const L = 48, n = L * L;
    // 低周波＋細かいノイズの初期場（拡散なら細部が落ちる）。
    const seed = new Float32Array(n);
    let s = 1;
    const rnd = (): number => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    for (let y = 0; y < L; y++)
      for (let x = 0; x < L; x++)
        seed[y * L + x] = 0.4 + 0.15 * Math.sin((6 * x) / L) + 0.2 * (rnd() - 0.5);
    // 高周波 RMS（隣接差＝細部の量）。
    const detail = (u: Float32Array): number => {
      let acc = 0;
      for (let y = 0; y < L; y++)
        for (let x = 0; x < L - 1; x++) acc += (u[y * L + x + 1] - u[y * L + x]) ** 2;
      return Math.sqrt(acc / n);
    };
    const d0 = detail(seed);
    const uDiff = seed.slice(), uLenia = seed.slice(), im = new Float32Array(n);
    for (let k = 0; k < 40; k++) leniaStep(uDiff, im, L, 0); // 拡散端
    for (let k = 0; k < 40; k++) leniaStep(uLenia, im, L, 1); // Lenia 端
    expect(detail(uDiff)).toBeLessThan(d0 * 0.5); // λ=0 は細部が落ちる（均す）
    expect(detail(uLenia)).toBeGreaterThan(detail(uDiff)); // λ=1 は構造が残る/立つ
  });

  it("既定（morph 省略）は λ=1 と一致（従来挙動を保つ）", () => {
    const L = 24, n = L * L;
    const a = new Float32Array(n), b = new Float32Array(n), im = new Float32Array(n);
    for (let i = 0; i < n; i++) { a[i] = (i % 7) / 7; b[i] = a[i]; }
    for (let k = 0; k < 8; k++) { leniaStep(a, im, L); leniaStep(b, im, L, 1); }
    for (let i = 0; i < n; i++) expect(a[i]).toBeCloseTo(b[i], 6);
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

describe("quatCglStep（四元数CGL・全色発展）", () => {
  const params = { b: 0.5, c: 0.9, D: 0, dt: 0.01 }; // D=0 で均質ODE（アトラクタ検証）

  it("|q| がアトラクタ 1 に収束（複素 |ψ|→1 の四元数版）", () => {
    const L = 4;
    const q = new Float32Array(L * L * 4);
    for (let i = 0; i < L * L; i++) {
      q[i * 4] = 0; // w=0 始動
      q[i * 4 + 1] = 0.3; q[i * 4 + 2] = 0.2; q[i * 4 + 3] = -0.1;
    }
    for (let s = 0; s < 800; s++) quatCglStep(q, L, params);
    for (let i = 0; i < L * L; i++) {
      const m = Math.hypot(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3]);
      expect(m).toBeGreaterThan(0.95);
      expect(m).toBeLessThan(1.05);
    }
  });

  it("純虚 3 成分すべてが時間発展する（明度 x も動く）", () => {
    const L = 4;
    const q = new Float32Array(L * L * 4);
    for (let i = 0; i < L * L; i++) { q[i * 4 + 1] = 0.3; q[i * 4 + 2] = 0.2; q[i * 4 + 3] = -0.1; }
    // 過渡後に各成分の可動を見る
    for (let s = 0; s < 300; s++) quatCglStep(q, L, params);
    const seen: number[][] = [[], [], []];
    for (let s = 0; s < 400; s++) {
      quatCglStep(q, L, params);
      seen[0].push(q[1]); seen[1].push(q[2]); seen[2].push(q[3]);
    }
    const std = (a: number[]): number => {
      const m = a.reduce((s, v) => s + v, 0) / a.length;
      return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
    };
    expect(std(seen[0])).toBeGreaterThan(0.05); // x(明度) が動く
    expect(std(seen[1])).toBeGreaterThan(0.05); // y(a) が動く
    expect(std(seen[2])).toBeGreaterThan(0.05); // z(b) が動く
  });
});

describe("quatCglStep coupling λ（scalar↔quat 代数結合ノブ）", () => {
  // 1 点の z 成分にインパルスを与え、他成分(w,x,y)への漏れ/z の変化 を測る（2 run 差分）。
  // checks/color-algebra.py と同型: λ=0 は完全独立(漏れ0)、λ=1 は結合(色相回転で漏れる)。
  function leak(coupling: number): number {
    const L = 24, n = L * L;
    const base = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      base[i * 4] = 0.1 * Math.sin(i * 0.3);
      base[i * 4 + 1] = 0.3 + 0.1 * Math.cos(i * 0.2);
      base[i * 4 + 2] = 0.2 + 0.1 * Math.sin(i * 0.17);
      base[i * 4 + 3] = -0.1 + 0.1 * Math.cos(i * 0.11);
    }
    const qa = base.slice(), qb = base.slice();
    const cIdx = (Math.floor(L / 2) * L + Math.floor(L / 2)) * 4;
    qb[cIdx + 3] += 0.5; // z にインパルス
    const p = { b: 0.6, c: 1.0, D: 4, dt: 0.01, coupling };
    for (let s = 0; s < 40; s++) { quatCglStep(qa, L, p); quatCglStep(qb, L, p); }
    let other = 0, zc = 0;
    for (let i = 0; i < n; i++) {
      other += Math.abs(qb[i * 4] - qa[i * 4]) +
        Math.abs(qb[i * 4 + 1] - qa[i * 4 + 1]) +
        Math.abs(qb[i * 4 + 2] - qa[i * 4 + 2]);
      zc += Math.abs(qb[i * 4 + 3] - qa[i * 4 + 3]);
    }
    return other / (zc + 1e-12);
  }

  it("λ=0 は成分が完全独立（z インパルスが他成分へ漏れない＝scalar 端）", () => {
    expect(leak(0)).toBeLessThan(1e-6);
  });

  it("λ=1 は結合（z インパルスが他成分へ漏れる＝色相回転が内蔵の quat 端）", () => {
    expect(leak(1)).toBeGreaterThan(0.1);
  });

  it("結合は λ とともに単調に増える（無段階ノブ）", () => {
    const l0 = leak(0), lh = leak(0.5), l1 = leak(1);
    expect(l0).toBeLessThan(lh);
    expect(lh).toBeLessThan(l1);
  });

  it("既定（coupling 省略）は λ=1 と一致（従来挙動を保つ）", () => {
    const L = 8, n = L * L;
    const q1 = new Float32Array(n * 4), q2 = new Float32Array(n * 4);
    for (let i = 0; i < n * 4; i++) { q1[i] = Math.sin(i); q2[i] = q1[i]; }
    const p = { b: 0.5, c: 0.9, D: 2, dt: 0.01 };
    for (let s = 0; s < 10; s++) {
      quatCglStep(q1, L, p); // coupling 省略
      quatCglStep(q2, L, { ...p, coupling: 1 });
    }
    for (let i = 0; i < n * 4; i++) expect(q1[i]).toBeCloseTo(q2[i], 6);
  });
});

// 写真代わりの滑らかな初期場。
function photoField(L: number): { re: Float32Array; im: Float32Array } {
  const re = new Float32Array(L * L);
  const im = new Float32Array(L * L);
  for (let y = 0; y < L; y++)
    for (let x = 0; x < L; x++) re[y * L + x] = 0.5 + 0.3 * Math.sin((6 * x) / L) - 0.5;
  return { re, im };
}
const bounded = (a: Float32Array, lim: number): boolean => a.every((v) => Number.isFinite(v) && Math.abs(v) < lim);
const std = (a: Float32Array): number => {
  const m = a.reduce((s, v) => s + v, 0) / a.length;
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
};

describe("waveStep（波動）", () => {
  it("有界に伝播（発散しない）", () => {
    const L = 24;
    const { re, im } = photoField(L);
    for (let s = 0; s < 500; s++) waveStep(re, im, L);
    expect(bounded(re, 3)).toBe(true);
    expect(std(re)).toBeGreaterThan(0); // 波が立っている
  });
});

describe("swiftHohenbergStep（縞）", () => {
  it("有界・パターンに自己組織化（分散が育つ）", () => {
    const L = 32;
    const { re, im } = photoField(L);
    for (let i = 0; i < re.length; i++) re[i] *= 0.2;
    for (let s = 0; s < 2000; s++) swiftHohenbergStep(re, im, L);
    expect(bounded(re, 2)).toBe(true);
    expect(std(re)).toBeGreaterThan(0.2); // ±1 縞へ
    expect(im.every((v) => v === 0)).toBe(true);
  });
});

describe("fitzHughNagumoStep（興奮性）", () => {
  it("有界（発散しない）", () => {
    const L = 24;
    const { re, im } = photoField(L);
    for (let i = 0; i < re.length; i++) re[i] *= 2;
    for (let s = 0; s < 800; s++) fitzHughNagumoStep(re, im, L);
    expect(bounded(re, 3)).toBe(true);
  });
});

describe("cahnHilliardStep（相分離）", () => {
  it("有界・±へ相分離（壁境界のため質量は近似保存）", () => {
    const L = 32;
    const { re, im } = photoField(L);
    for (let i = 0; i < re.length; i++) re[i] *= 0.4;
    for (let s = 0; s < 3000; s++) cahnHilliardStep(re, im, L);
    expect(bounded(re, 2)).toBe(true); // 発散しない
    expect(std(re)).toBeGreaterThan(0.3); // ±ドメインへ分離
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
