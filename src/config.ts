// アプリ設定：グリッド・CGL パラメータ・モード・ブレンド・種。
// JSON / URL に直列化して同一作品を再現できる（書き出し設計の 1 項目）。

import { clampParams, DEFAULT_PARAMS, type CGLParams } from "./engine/params";
import type { Seed } from "./image/imageToField";

// A=写真を流す(flow) / B=場を OKLCh で表示(field) / blend=A↔B を混合。
export type Mode = "A" | "B" | "blend";

// 力学エンジン。mode(A/B/blend) は cgl のみ有効。
// cgl=複素GL / grayscott=反応拡散(Turing) / lenia=連続ライフ / chroma=色拡散 /
// quat=四元数(全色発展) / scalar=実数スカラー(全色・独立) / unified=統合形(四元数・CGL↔SH) /
// telegraph=電信方程式(拡散↔波動の統一) / swifthohenberg=縞・六方 /
// fitzhugh=興奮性(伝播波) / cahnhilliard=相分離。
export type Dynamics =
  | "cgl" | "grayscott" | "lenia" | "chroma" | "quat" | "scalar" | "unified"
  | "telegraph" | "swifthohenberg" | "fitzhugh" | "cahnhilliard";

export interface AppConfig {
  L: number;
  params: CGLParams;
  mode: Mode;
  blend: number; // 0=A .. 1=B（mode="blend" のときだけ効く）
  seed: Seed; // 複素種の作り方
  dynamics: Dynamics;
}

export const DEFAULT_CONFIG: AppConfig = {
  L: 256,
  params: DEFAULT_PARAMS,
  mode: "A",
  blend: 0.5,
  seed: "color", // 写真の色をそのまま場に（色相→位相・彩度→振幅）
  dynamics: "cgl",
};

const MODES: readonly Mode[] = ["A", "B", "blend"];
const SEEDS: readonly Seed[] = ["color", "phase", "amp"];
const DYNAMICS: readonly Dynamics[] = [
  "cgl", "grayscott", "lenia", "chroma", "quat", "scalar", "unified",
  "telegraph", "swifthohenberg", "fitzhugh", "cahnhilliard",
];

function clampL(v: unknown): number {
  const n = typeof v === "number" ? v : NaN;
  if (!Number.isFinite(n)) return DEFAULT_CONFIG.L;
  return Math.round(Math.min(1024, Math.max(64, n)));
}

function clamp01(v: unknown, fallback: number): number {
  const n = typeof v === "number" ? v : NaN;
  return Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : fallback;
}

// 任意の（信頼できない）入力を安全な AppConfig に丸める。
export function normalizeConfig(raw: Partial<AppConfig> | null | undefined): AppConfig {
  const r = raw ?? {};
  return {
    L: clampL(r.L),
    params: clampParams(r.params),
    mode: MODES.includes(r.mode as Mode) ? (r.mode as Mode) : DEFAULT_CONFIG.mode,
    blend: clamp01(r.blend, DEFAULT_CONFIG.blend),
    seed: SEEDS.includes(r.seed as Seed) ? (r.seed as Seed) : DEFAULT_CONFIG.seed,
    dynamics: DYNAMICS.includes(r.dynamics as Dynamics) ? (r.dynamics as Dynamics) : DEFAULT_CONFIG.dynamics,
  };
}

export function encodeConfig(c: AppConfig): string {
  return JSON.stringify(c);
}

export function decodeConfig(json: string): AppConfig {
  let raw: unknown = null;
  try {
    raw = JSON.parse(json);
  } catch {
    raw = null;
  }
  return normalizeConfig(raw as Partial<AppConfig> | null);
}

// --- URL 直列化（?c=<base64url>） -------------------------------------------

function b64urlEncode(s: string): string {
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlDecode(s: string): string {
  return atob(s.replace(/-/g, "+").replace(/_/g, "/"));
}

export function encodeConfigURL(c: AppConfig): string {
  return b64urlEncode(encodeConfig(c));
}

export function decodeConfigURL(s: string): AppConfig {
  try {
    return decodeConfig(b64urlDecode(s));
  } catch {
    return DEFAULT_CONFIG;
  }
}

// --- プリセット -------------------------------------------------------------

export const PRESETS: readonly { name: string; config: AppConfig }[] = [
  {
    name: "螺旋波・写真を流す (A)",
    config: normalizeConfig({ mode: "A", seed: "phase", params: { ...DEFAULT_PARAMS, b: 0.5, c: 0.5 } }),
  },
  {
    name: "写真の色で模様 (B)",
    config: normalizeConfig({ mode: "B", seed: "color", params: { ...DEFAULT_PARAMS, b: 2, c: -1, dt: 0.01 } }),
  },
  {
    name: "溶けていく (blend)",
    config: normalizeConfig({ mode: "blend", blend: 0.3, seed: "color", params: { ...DEFAULT_PARAMS } }),
  },
];
