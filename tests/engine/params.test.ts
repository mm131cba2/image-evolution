import { describe, it, expect } from "vitest";
import {
  clampParams,
  DEFAULT_PARAMS,
  PARAM_RANGES,
  isBenjaminFeirStable,
} from "../../src/engine/params";

describe("clampParams", () => {
  it("空入力は既定値を返す", () => {
    expect(clampParams()).toEqual(DEFAULT_PARAMS);
    expect(clampParams({})).toEqual(DEFAULT_PARAMS);
  });

  it("範囲外は境界へ丸める", () => {
    const c = clampParams({ b: 99, c: -99, D: 0, dt: 10, speed: -5 });
    expect(c.b).toBe(PARAM_RANGES.b[1]); // 3
    expect(c.c).toBe(PARAM_RANGES.c[0]); // -3
    expect(c.D).toBe(PARAM_RANGES.D[0]); // 0.5
    expect(c.dt).toBe(PARAM_RANGES.dt[1]); // 0.05
    expect(c.speed).toBe(PARAM_RANGES.speed[0]); // 0
  });

  it("範囲内はそのまま通す", () => {
    const p = { b: 1.2, c: -0.8, D: 6, dt: 0.01, speed: 2 };
    expect(clampParams(p)).toEqual(p);
  });

  it("NaN / Infinity / 非数は既定値へ落とす", () => {
    const c = clampParams({
      b: NaN,
      c: Infinity,
      D: -Infinity,
      // @ts-expect-error 実行時の不正入力（JSON 由来）を模す
      dt: "0.02",
      speed: NaN,
    });
    expect(c.b).toBe(DEFAULT_PARAMS.b);
    expect(c.c).toBe(DEFAULT_PARAMS.c);
    expect(c.D).toBe(DEFAULT_PARAMS.D);
    expect(c.dt).toBe(DEFAULT_PARAMS.dt);
    expect(c.speed).toBe(DEFAULT_PARAMS.speed);
  });

  it("部分入力は欠損だけ既定で埋める", () => {
    const c = clampParams({ b: 2 });
    expect(c.b).toBe(2);
    expect(c.c).toBe(DEFAULT_PARAMS.c);
    expect(c.D).toBe(DEFAULT_PARAMS.D);
  });
});

describe("既定パラメータ", () => {
  it("Benjamin-Feir 安定（1+bc>0）である", () => {
    expect(1 + DEFAULT_PARAMS.b * DEFAULT_PARAMS.c).toBeGreaterThan(0);
    expect(isBenjaminFeirStable(DEFAULT_PARAMS)).toBe(true);
  });

  it("乱流パラメータは不安定判定になる", () => {
    expect(isBenjaminFeirStable({ b: 2, c: -1, D: 4, dt: 0.01, speed: 1 })).toBe(false);
  });
});
