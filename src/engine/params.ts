// CGL（複素 Ginzburg-Landau）の力学パラメータと、UI/JSON 入力を安全域へ丸めるクランプ。
//
// 方程式: ∂ψ/∂t = ψ + (1+ib)·D·∇²ψ − (1+ic)·|ψ|²·ψ
//   b, c … 力学モード（積 1+bc の符号が Benjamin-Feir 判定: >0 安定・<0 位相乱流）
//   D    … 拡散係数（構造スケール。既定 4 は検証値 checks/cgl-explicit.py 相当）
//   dt   … 時間刻み（実空間陽解法では dt≈0.01 で乱流も安定・dt=0.02 は安定域のみ）
//   speed… モード A の流速（変位場移流の強さ・0 で静止）

export interface CGLParams {
  b: number;
  c: number;
  D: number;
  dt: number;
  speed: number;
}

// 既定は Benjamin-Feir 安定（1+bc = 1.25 > 0）＝きれいな螺旋波。
export const DEFAULT_PARAMS: CGLParams = { b: 0.5, c: 0.5, D: 4, dt: 0.02, speed: 1 };

// 各パラメータの許容範囲 [min, max]。
export const PARAM_RANGES = {
  b: [-3, 3],
  c: [-3, 3],
  D: [0.5, 16],
  dt: [1e-3, 0.05],
  speed: [0, 4],
} as const satisfies Record<keyof CGLParams, readonly [number, number]>;

function clamp1(v: unknown, lo: number, hi: number, fallback: number): number {
  const n = typeof v === "number" ? v : NaN;
  if (!Number.isFinite(n)) return fallback; // NaN/Infinity/非数は既定へ
  return Math.min(hi, Math.max(lo, n));
}

// 部分入力（UI スライダー・JSON・URL 復元）を安全な CGLParams に丸める。
// 欠損・NaN・範囲外は既定値/境界へ寄せる（発散や不正状態を防ぐ）。
export function clampParams(p: Partial<CGLParams> = {}): CGLParams {
  return {
    b: clamp1(p.b, PARAM_RANGES.b[0], PARAM_RANGES.b[1], DEFAULT_PARAMS.b),
    c: clamp1(p.c, PARAM_RANGES.c[0], PARAM_RANGES.c[1], DEFAULT_PARAMS.c),
    D: clamp1(p.D, PARAM_RANGES.D[0], PARAM_RANGES.D[1], DEFAULT_PARAMS.D),
    dt: clamp1(p.dt, PARAM_RANGES.dt[0], PARAM_RANGES.dt[1], DEFAULT_PARAMS.dt),
    speed: clamp1(p.speed, PARAM_RANGES.speed[0], PARAM_RANGES.speed[1], DEFAULT_PARAMS.speed),
  };
}

// Benjamin-Feir 安定か（1+bc>0）。UI のガイド表示に使う。
export function isBenjaminFeirStable(p: CGLParams): boolean {
  return 1 + p.b * p.c > 0;
}
