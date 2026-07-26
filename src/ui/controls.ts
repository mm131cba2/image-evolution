// UI コントロール（CGL パラメータのスライダー＋画像選択＋種＋一時停止）。
// DOM 構築は buildControls。純ロジック（対数スケール・安定判定）は分離してテストする。

import type { CGLParams } from "../engine/params";
import { PARAM_RANGES } from "../engine/params";
import type { Seed } from "../image/imageToField";
import type { Mode, Dynamics } from "../config";

// 対数スケール: t∈[0,1] → [lo,hi]（dt のように桁が広い量に）。
export function logScale(t: number, lo: number, hi: number): number {
  return lo * (hi / lo) ** Math.min(1, Math.max(0, t));
}
export function logScaleInv(v: number, lo: number, hi: number): number {
  return Math.log(v / lo) / Math.log(hi / lo);
}

// Benjamin-Feir: 1+bc>0 で安定（螺旋波）・<0 で位相乱流。
export function stabilityText(p: CGLParams): string {
  const s = 1 + p.b * p.c;
  return s > 0 ? `安定・螺旋波 (1+bc=${s.toFixed(2)})` : `位相乱流 (1+bc=${s.toFixed(2)})`;
}

// 乱流域で dt が大きいと実空間陽解法が過増幅する（dt≤0.01 が必要・checks/cgl-explicit.py）。
export function needsSmallerDt(p: CGLParams): boolean {
  return 1 + p.b * p.c < 0 && p.dt > 0.012;
}

// 拡散の陽解法 CFL: dt·D·√(1+b²)·8 が ~2 を超えると最高周波数モードが発散する。
// D を上げる／dt を上げると無警告で NaN 化するのを防ぐ（数値実測: D=12,dt=0.02 で発散）。
export function diffusionUnstable(p: CGLParams): boolean {
  return p.dt * p.D * Math.hypot(1, p.b) * 8 > 1.8;
}

export interface ControlCallbacks {
  onParams: (p: CGLParams) => void;
  onMode: (m: Mode) => void;
  onBlend: (v: number) => void;
  onReset: () => void;
  onFile: (f: File) => void;
  onSeedType: (s: Seed) => void;
  onDynamics: (d: Dynamics) => void;
  onCoRotate: (on: boolean) => void;
  onAnchor: (v: number) => void;
  onYRate: (v: number) => void;
  onGamma: (v: number) => void;
  onRecord: (seconds: number, onTick: (remaining: number) => void) => Promise<void>;
  onTogglePause: () => boolean; // 戻り値: 再生中か
}

function row(label: string): { row: HTMLDivElement; value: HTMLSpanElement } {
  const r = document.createElement("div");
  r.style.cssText = "display:flex;align-items:center;gap:8px;margin:2px 0;";
  const l = document.createElement("span");
  l.textContent = label;
  l.style.cssText = "width:2.5em;text-align:right;";
  const value = document.createElement("span");
  value.style.cssText = "width:4em;color:#9cf;font-variant-numeric:tabular-nums;";
  r.appendChild(l);
  return { row: r, value };
}

function linSlider(
  label: string,
  key: "b" | "c" | "D" | "speed",
  p: CGLParams,
  onInput: (v: number) => void,
): HTMLDivElement {
  const [lo, hi] = PARAM_RANGES[key];
  const { row: r, value } = row(label);
  const input = document.createElement("input");
  input.type = "range";
  input.min = String(lo);
  input.max = String(hi);
  input.step = "0.01";
  input.value = String(p[key]);
  input.style.flex = "1";
  value.textContent = p[key].toFixed(2);
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    value.textContent = v.toFixed(2);
    onInput(v);
  });
  r.appendChild(input);
  r.appendChild(value);
  return r;
}

export function buildControls(
  initial: CGLParams,
  initialMode: Mode,
  initialBlend: number,
  cb: ControlCallbacks,
): void {
  const p: CGLParams = { ...initial };
  const panel = document.createElement("div");
  panel.style.cssText =
    "position:fixed;top:10px;left:10px;width:260px;font-family:sans-serif;font-size:13px;" +
    "color:#ddd;background:rgba(0,0,0,.55);padding:10px 12px;border-radius:10px;";

  // 力学ごとの可視性制御: 各行に「どの力学で出すか」を登録し、選択で切り替える。
  let curDyn: Dynamics = "cgl";
  let curMode: Mode = initialMode;
  const tracked: { el: HTMLElement; shown: string; show: (d: Dynamics, m: Mode) => boolean }[] = [];
  const track = (el: HTMLElement, show: (d: Dynamics, m: Mode) => boolean): void => {
    tracked.push({ el, shown: el.style.display, show });
  };
  const updateVis = (): void => {
    for (const t of tracked) t.el.style.display = t.show(curDyn, curMode) ? t.shown : "none";
  };
  const isCgl = (d: Dynamics): boolean => d === "cgl";
  const usesBC = (d: Dynamics): boolean => d === "cgl" || d === "quat"; // b,c を使う
  const usesDiff = (d: Dynamics): boolean => d === "cgl" || d === "chroma" || d === "quat"; // D,dt

  // 力学エンジン選択（最上位・常時）。
  const dynRow = document.createElement("div");
  dynRow.style.cssText = "display:flex;gap:8px;align-items:center;margin-bottom:6px;";
  const dynSel = document.createElement("select");
  for (const [v, t] of [
    ["cgl", "力学: 複素GL（螺旋波・流れ）"],
    ["grayscott", "力学: 反応拡散（Turing斑点）"],
    ["lenia", "力学: Lenia（連続ライフ）"],
    ["chroma", "力学: 色拡散（YCbCr）"],
    ["quat", "力学: 四元数（全色発展）"],
    ["telegraph", "力学: 電信方程式（拡散↔波動）"],
    ["swifthohenberg", "力学: Swift-Hohenberg（縞）"],
    ["fitzhugh", "力学: FitzHugh-Nagumo（興奮性）"],
    ["cahnhilliard", "力学: Cahn-Hilliard（相分離）"],
  ] as const) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    dynSel.appendChild(o);
  }
  dynSel.style.flex = "1";
  dynSel.addEventListener("change", () => {
    curDyn = dynSel.value as Dynamics;
    cb.onDynamics(curDyn);
    updateVis();
  });
  dynRow.appendChild(dynSel);
  panel.appendChild(dynRow);

  // 調整項目のない力学（gs/lenia）向けのヒント。
  const hint = document.createElement("div");
  hint.style.cssText = "color:#9a9;font-size:12px;margin:2px 0;";
  hint.textContent = "このモードは調整項目がありません（固定パラメータ）";
  panel.appendChild(hint);
  const fixedParamDyn = (d: Dynamics): boolean =>
    d === "grayscott" || d === "lenia" || d === "swifthohenberg" ||
    d === "fitzhugh" || d === "cahnhilliard";
  track(hint, fixedParamDyn);

  // モード選択（cgl のみ・A=写真を流す / B=場を表示 / blend=混合）。
  const modeRow = document.createElement("div");
  modeRow.style.cssText = "display:flex;gap:8px;align-items:center;margin-bottom:6px;";
  const modeSel = document.createElement("select");
  for (const [v, t] of [
    ["A", "A: 写真を流す"],
    ["B", "B: 場を表示"],
    ["blend", "blend: 混合"],
  ] as const) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    if (v === initialMode) o.selected = true;
    modeSel.appendChild(o);
  }
  modeSel.style.flex = "1";
  modeSel.addEventListener("change", () => {
    curMode = modeSel.value as Mode;
    cb.onMode(curMode);
    updateVis();
  });
  modeRow.appendChild(modeSel);
  panel.appendChild(modeRow);
  track(modeRow, isCgl);

  // blend スライダー（cgl かつ mode=blend のときだけ）。
  {
    const { row: r, value } = row("blend");
    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = "1";
    input.step = "0.01";
    input.value = String(initialBlend);
    input.style.flex = "1";
    value.textContent = initialBlend.toFixed(2);
    input.addEventListener("input", () => {
      const v = parseFloat(input.value);
      value.textContent = v.toFixed(2);
      cb.onBlend(v);
    });
    r.appendChild(input);
    r.appendChild(value);
    panel.appendChild(r);
    track(r, (d, m) => d === "cgl" && m === "blend");
  }

  const status = document.createElement("div");
  status.style.cssText = "margin-bottom:6px;color:#bbb;";
  const warn = document.createElement("div");
  warn.style.cssText = "color:#fc6;font-size:12px;min-height:1em;";
  const refresh = () => {
    status.textContent = stabilityText(p);
    warn.textContent = diffusionUnstable(p)
      ? "⚠ D×dt が大きすぎて発散します（D か dt を下げる）"
      : needsSmallerDt(p)
        ? "⚠ 乱流は dt≤0.01 推奨（過増幅を防ぐ）"
        : "";
    cb.onParams({ ...p });
  };

  panel.appendChild(status);
  track(status, usesBC); // 1+bc は b,c を使う力学だけ

  // b,c,D,流速 スライダー（適用範囲ごとに表示）。
  const sliderShow: Record<string, (d: Dynamics) => boolean> = {
    b: usesBC, c: usesBC, D: usesDiff, speed: isCgl,
  };
  for (const [lbl, key] of [
    ["b", "b"],
    ["c", "c"],
    ["D", "D"],
    ["流速", "speed"],
  ] as const) {
    const r = linSlider(lbl, key, p, (v) => {
      p[key] = v;
      refresh();
    });
    panel.appendChild(r);
    track(r, sliderShow[key]);
  }

  // dt は対数スライダー。
  {
    const [lo, hi] = PARAM_RANGES.dt;
    const { row: r, value } = row("dt");
    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = "1";
    input.step = "0.001";
    input.value = String(logScaleInv(p.dt, lo, hi));
    input.style.flex = "1";
    value.textContent = p.dt.toFixed(4);
    input.addEventListener("input", () => {
      p.dt = logScale(parseFloat(input.value), lo, hi);
      value.textContent = p.dt.toFixed(4);
      refresh();
    });
    r.appendChild(input);
    r.appendChild(value);
    panel.appendChild(r);
    track(r, usesDiff);
  }
  panel.appendChild(warn);
  track(warn, usesDiff);

  // 種の種類（cgl のみ）。
  const seedRow = document.createElement("div");
  seedRow.style.cssText = "display:flex;gap:8px;align-items:center;margin-top:6px;";
  const seedSel = document.createElement("select");
  for (const [v, t] of [
    ["color", "種: 写真の色（色相・彩度）"],
    ["phase", "種: 明るさ→位相"],
    ["amp", "種: 明るさ→振幅"],
  ] as const) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    seedSel.appendChild(o);
  }
  seedSel.style.flex = "1";
  seedSel.addEventListener("change", () => cb.onSeedType(seedSel.value as Seed));
  seedRow.appendChild(seedSel);
  panel.appendChild(seedRow);
  track(seedRow, isCgl);

  // 色の全画面ストロボ（一様位相回転）を止める（cgl の色表示・quat）。既定 ON。
  const coRow = document.createElement("label");
  coRow.style.cssText = "display:flex;gap:6px;align-items:center;margin-top:6px;cursor:pointer;";
  const coChk = document.createElement("input");
  coChk.type = "checkbox";
  coChk.checked = true;
  coChk.addEventListener("change", () => cb.onCoRotate(coChk.checked));
  coRow.appendChild(coChk);
  coRow.appendChild(document.createTextNode("色の回転を止める"));
  panel.appendChild(coRow);
  track(coRow, usesBC);

  // 四元数モードの「元の色味を保つ」強度（色重心を写真に引き戻す・0=自由発展）。
  {
    const { row: r, value } = row("元色");
    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = "1";
    input.step = "0.01";
    input.value = "0.6";
    input.style.flex = "1";
    value.textContent = "0.60";
    input.addEventListener("input", () => {
      const v = parseFloat(input.value);
      value.textContent = v.toFixed(2);
      cb.onAnchor(v);
    });
    r.appendChild(input);
    r.appendChild(value);
    panel.appendChild(r);
    track(r, (d) => d === "quat");
  }

  // 色拡散(chroma)の輝度拡散率（0=Y固定=形保持・>0=形も溶ける）。
  {
    const { row: r, value } = row("輝度");
    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = "1";
    input.step = "0.01";
    input.value = "0.4";
    input.style.flex = "1";
    value.textContent = "0.40";
    input.addEventListener("input", () => {
      const v = parseFloat(input.value);
      value.textContent = v.toFixed(2);
      cb.onYRate(v);
    });
    r.appendChild(input);
    r.appendChild(value);
    panel.appendChild(r);
    track(r, (d) => d === "chroma");
  }

  // 電信方程式の減衰 γ（対数・小=波動 / 大=拡散）。拡散↔波動の統一ツマミ。
  {
    const [lo, hi] = [0.002, 3];
    const { row: r, value } = row("減衰");
    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = "1";
    input.step = "0.001";
    input.value = String(logScaleInv(0.05, lo, hi));
    input.style.flex = "1";
    value.textContent = "0.05";
    input.addEventListener("input", () => {
      const g = logScale(parseFloat(input.value), lo, hi);
      value.textContent = g.toFixed(3);
      cb.onGamma(g);
    });
    r.appendChild(input);
    r.appendChild(value);
    panel.appendChild(r);
    track(r, (d) => d === "telegraph");
  }

  // ボタン列: 画像・リセット・一時停止・録画（常時）。
  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;";

  const fileLabel = document.createElement("label");
  fileLabel.textContent = "画像を選ぶ";
  fileLabel.style.cssText = "cursor:pointer;text-decoration:underline;";
  const file = document.createElement("input");
  file.type = "file";
  file.accept = "image/*";
  file.style.display = "none";
  file.addEventListener("change", () => {
    const f = file.files?.[0];
    if (f) cb.onFile(f);
  });
  fileLabel.appendChild(file);

  const reset = document.createElement("button");
  reset.textContent = "↻ リセット";
  reset.style.cursor = "pointer";
  reset.addEventListener("click", () => cb.onReset());

  const pause = document.createElement("button");
  pause.textContent = "⏸";
  pause.style.cursor = "pointer";
  pause.addEventListener("click", () => {
    pause.textContent = cb.onTogglePause() ? "⏸" : "▶";
  });

  // 録画: WebM をダウンロード（秒数は隣の入力・既定 8）。
  const rec = document.createElement("button");
  rec.textContent = "⏺ 録画";
  rec.style.cursor = "pointer";
  const secInput = document.createElement("input");
  secInput.type = "number";
  secInput.min = "1";
  secInput.max = "60";
  secInput.value = "8";
  secInput.title = "録画秒数";
  secInput.style.cssText = "width:3em;";
  rec.addEventListener("click", () => {
    if (rec.disabled) return;
    rec.disabled = true;
    const secs = Math.min(60, Math.max(1, parseInt(secInput.value) || 8));
    void cb
      .onRecord(secs, (remaining) => { rec.textContent = `⏺ ${remaining}s`; })
      .finally(() => { rec.textContent = "⏺ 録画"; rec.disabled = false; });
  });

  btnRow.appendChild(fileLabel);
  btnRow.appendChild(reset);
  btnRow.appendChild(pause);
  btnRow.appendChild(rec);
  btnRow.appendChild(secInput);
  panel.appendChild(btnRow);

  document.body.appendChild(panel);
  refresh();
  updateVis(); // 初期は cgl の項目だけ表示
}
