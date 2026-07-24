import { describe, it, expect } from "vitest";
import { logScale, logScaleInv, stabilityText, needsSmallerDt } from "../../src/ui/controls";
import type { CGLParams } from "../../src/engine/params";

const P = (over: Partial<CGLParams> = {}): CGLParams => ({
  b: 0.5,
  c: 0.5,
  D: 4,
  dt: 0.02,
  speed: 1,
  ...over,
});

describe("logScale", () => {
  it("端点を保つ", () => {
    expect(logScale(0, 1e-3, 0.05)).toBeCloseTo(1e-3, 8);
    expect(logScale(1, 1e-3, 0.05)).toBeCloseTo(0.05, 8);
  });
  it("往復で一致", () => {
    for (const v of [1e-3, 0.005, 0.01, 0.02, 0.05]) {
      expect(logScale(logScaleInv(v, 1e-3, 0.05), 1e-3, 0.05)).toBeCloseTo(v, 8);
    }
  });
  it("中点は幾何平均（対数中央）", () => {
    expect(logScale(0.5, 1e-3, 0.1)).toBeCloseTo(0.01, 6); // sqrt(1e-3*0.1)=0.01
  });
});

describe("stabilityText", () => {
  it("1+bc>0 は安定・<0 は乱流", () => {
    expect(stabilityText(P({ b: 0.5, c: 0.5 }))).toContain("安定");
    expect(stabilityText(P({ b: 2, c: -1 }))).toContain("乱流");
  });
});

describe("needsSmallerDt", () => {
  it("乱流かつ dt>0.012 で警告", () => {
    expect(needsSmallerDt(P({ b: 2, c: -1, dt: 0.02 }))).toBe(true);
    expect(needsSmallerDt(P({ b: 2, c: -1, dt: 0.01 }))).toBe(false); // dt 十分小
    expect(needsSmallerDt(P({ b: 0.5, c: 0.5, dt: 0.02 }))).toBe(false); // 安定域
  });
});
