import { describe, it, expect } from "vitest";
import {
  DEFAULT_CONFIG,
  normalizeConfig,
  encodeConfig,
  decodeConfig,
  encodeConfigURL,
  decodeConfigURL,
  PRESETS,
} from "../src/config";

describe("normalizeConfig", () => {
  it("空/欠損は既定へ", () => {
    expect(normalizeConfig(null)).toEqual(DEFAULT_CONFIG);
    expect(normalizeConfig({})).toEqual(DEFAULT_CONFIG);
  });

  it("不正な mode/seed は既定へ・blend と L はクランプ", () => {
    const c = normalizeConfig({
      // @ts-expect-error 不正入力
      mode: "X",
      // @ts-expect-error 不正入力
      seed: "zzz",
      blend: 5,
      L: 10,
    });
    expect(c.mode).toBe(DEFAULT_CONFIG.mode);
    expect(c.seed).toBe(DEFAULT_CONFIG.seed);
    expect(c.blend).toBe(1); // 5 → クランプ 1
    expect(c.L).toBe(64); // 10 → 下限 64
  });

  it("params は clampParams を通す", () => {
    const c = normalizeConfig({ params: { b: 99, c: 0, D: 4, dt: 0.02, speed: 1 } });
    expect(c.params.b).toBe(3); // 99 → 上限 3
  });
});

describe("直列化の往復", () => {
  const cfg = normalizeConfig({ mode: "blend", blend: 0.3, L: 128, params: { b: 1.5, c: -0.5, D: 8, dt: 0.01, speed: 2 } });

  it("JSON 往復で一致", () => {
    expect(decodeConfig(encodeConfig(cfg))).toEqual(cfg);
  });

  it("URL 往復で一致", () => {
    expect(decodeConfigURL(encodeConfigURL(cfg))).toEqual(cfg);
  });

  it("壊れた JSON / URL は既定へ", () => {
    expect(decodeConfig("{not json")).toEqual(DEFAULT_CONFIG);
    expect(decodeConfigURL("!!!not base64!!!")).toEqual(DEFAULT_CONFIG);
  });
});

describe("PRESETS", () => {
  it("すべて正規化済み（normalize が恒等）", () => {
    for (const p of PRESETS) {
      expect(normalizeConfig(p.config)).toEqual(p.config);
      expect(p.name.length).toBeGreaterThan(0);
    }
  });

  it("乱流プリセットは dt<=0.01（陽解法の安定域）", () => {
    const turb = PRESETS.find((p) => p.name.includes("乱流"));
    expect(turb).toBeDefined();
    expect(turb!.config.params.dt).toBeLessThanOrEqual(0.01);
  });
});
