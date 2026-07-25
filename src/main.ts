// エントリ: WebGPU 初期化 → CGL エンジン → 種投入 → RAF ループ。
// フェーズ1 の最小形（モード B 表示）。まず既定の種で螺旋波を見せ、画像を選べば
// その輝度を位相に種付けして時間発展させる。

import { initWebGPU } from "./engine/gpu";
import { CGLEngine, type ModeNum, type DynNum } from "./engine/cglGPU";
import { buildCyclicLUT } from "./render/palette";
import { DEFAULT_CONFIG } from "./config";
import {
  imageToField,
  seedGrayScott,
  seedLenia,
  seedChroma,
  type Seed,
  type Field,
} from "./image/imageToField";
import { buildControls } from "./ui/controls";
import type { CGLParams } from "./engine/params";
import type { Mode, Dynamics } from "./config";

const L = DEFAULT_CONFIG.L; // 256

function modeNum(m: Mode): ModeNum {
  return m === "A" ? 0 : m === "B" ? 1 : 2;
}

function dynNum(d: Dynamics): DynNum {
  return d === "cgl" ? 0 : d === "grayscott" ? 1 : d === "lenia" ? 2 : 3;
}

// 画像未選択でもモード A が何か映すための既定原本（滑らかな色のグラデ・線形光 RGBA）。
function defaultOriginal(): Float32Array {
  const orig = new Float32Array(L * L * 4);
  for (let y = 0; y < L; y++) {
    for (let x = 0; x < L; x++) {
      const i = (y * L + x) * 4;
      orig[i] = 0.5 + 0.4 * Math.sin((2 * Math.PI * x) / L);
      orig[i + 1] = 0.5 + 0.4 * Math.sin((2 * Math.PI * y) / L + 1.7);
      orig[i + 2] = 0.5 + 0.4 * Math.sin((2 * Math.PI * (x + y)) / L + 3.1);
      orig[i + 3] = 1;
    }
  }
  return orig;
}

function overlay(msg: string): void {
  const d = document.createElement("div");
  d.style.cssText =
    "position:fixed;inset:0;display:flex;align-items:center;justify-content:center;" +
    "color:#ccc;font-family:sans-serif;font-size:16px;padding:2rem;text-align:center;";
  d.textContent = msg;
  document.body.appendChild(d);
}

// 画像ファイル → Field（原本 RGBA + 複素種・面積平均は imageToField 側）。
async function fieldFromFile(file: File, seed: Seed): Promise<Field> {
  const bmp = await createImageBitmap(file);
  const cap = 2048;
  const scale = Math.min(1, cap / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * scale));
  const h = Math.max(1, Math.round(bmp.height * scale));
  const oc = document.createElement("canvas");
  oc.width = w;
  oc.height = h;
  const c2 = oc.getContext("2d");
  if (!c2) throw new Error("2d context 取得失敗");
  c2.drawImage(bmp, 0, 0, w, h);
  const id = c2.getImageData(0, 0, w, h);
  return imageToField(id, L, seed);
}

async function main(): Promise<void> {
  const canvas = document.getElementById("app") as HTMLCanvasElement | null;
  if (!canvas) return;
  canvas.width = L;
  canvas.height = L;

  const gpu = await initWebGPU(canvas);
  if (!gpu) {
    overlay("WebGPU 対応ブラウザが必要です（Chrome/Edge、Safari 26+、Firefox 最新）。");
    return;
  }

  const engine = new CGLEngine(gpu.device, gpu.format, L, buildCyclicLUT());

  // 表示状態（既定は安全に動く B。画像を入れて A に切り替えると写真が流れる）。
  const params: CGLParams = { ...DEFAULT_CONFIG.params };
  let mode: Mode = "B";
  let blend = 0.5;
  let seedType: Seed = DEFAULT_CONFIG.seed;
  let dynamics: Dynamics = DEFAULT_CONFIG.dynamics;
  let lastFile: File | null = null;
  let running = true;
  // 共回転フレーム: CGL の一様位相回転 dθ/dt=−c を表示上で打ち消す（全画面の色ストロボ防止）。
  let coRotate = true;
  let phaseRef = 0;

  const apply = (): void => engine.setState(params, modeNum(mode), blend, phaseRef);

  // 画像未選択の既定場（cgl はランダム位相・非cglは既定グラデを種に）。1度だけ生成し固定。
  const defaultField = (): Field => {
    const orig = defaultOriginal();
    const psiRe = new Float32Array(L * L);
    const psiIm = new Float32Array(L * L);
    for (let i = 0; i < L * L; i++) {
      const th = Math.random() * 2 * Math.PI;
      psiRe[i] = Math.cos(th);
      psiIm[i] = Math.sin(th);
    }
    return { orig, psiRe, psiIm, L };
  };
  const defaultF = defaultField();

  // 現在の力学に応じて engine に種を投入（写真か既定場の orig から状態を作る）。
  const seedFromField = (f: Field): void => {
    engine.seedOriginal(f.orig);
    if (dynamics === "cgl") {
      engine.seed(f.psiRe, f.psiIm);
    } else {
      const s =
        dynamics === "grayscott" ? seedGrayScott(f.orig, L)
        : dynamics === "lenia" ? seedLenia(f.orig, L)
        : seedChroma(f.orig, L);
      engine.seed(s.re, s.im);
    }
    engine.resetMap();
    phaseRef = 0; // t=0 に戻す（共回転オフセットもリセット）
  };

  // 種を作り直す（写真は現 seedType で再取得・未選択は固定の既定場）。
  const rebuild = (): void => {
    if (lastFile) {
      void fieldFromFile(lastFile, seedType).then(seedFromField);
    } else {
      seedFromField(defaultF);
    }
  };

  engine.setDynamics(dynNum(dynamics));
  seedFromField(defaultF);
  apply();

  buildControls(DEFAULT_CONFIG.params, mode, blend, {
    onParams: (p) => {
      Object.assign(params, p);
      apply();
    },
    onMode: (m) => {
      mode = m;
      apply();
    },
    onBlend: (v) => {
      blend = v;
      apply();
    },
    onReset: () => {
      rebuild();
    },
    onFile: (file) => {
      lastFile = file;
      rebuild();
    },
    onSeedType: (s) => {
      seedType = s;
      rebuild(); // 種の作り方を変えたら作り直す（cgl のみ効く）
    },
    onDynamics: (d) => {
      dynamics = d;
      engine.setDynamics(dynNum(d));
      rebuild(); // 力学に応じて種を作り直す
    },
    onCoRotate: (on) => {
      coRotate = on;
    },
    onTogglePause: () => {
      running = !running;
      return running;
    },
  });

  // フレームレート非依存: 実時間を貯めて 1/60s ぶんの論理フレームだけ進める。
  // 60Hz では毎フレーム 1 回（従来と同一）・120Hz では 1 回おき・30Hz では 2 回進み、
  // 表示装置に依らず同じ速さで時間発展する（ループ書き出しの再現性にも効く）。
  const STEPS_PER_FRAME = 6;
  const LOGICAL_MS = 1000 / 60;
  const MAX_CATCHUP = 4; // 背景タブ復帰などの暴走を防ぐ上限
  let acc = 0;
  let last = performance.now();
  const advance = (): void => {
    engine.stepCGL(STEPS_PER_FRAME);
    if (coRotate) phaseRef += params.c * params.dt * STEPS_PER_FRAME; // +c·Δt で一様回転を相殺
    // 写真の移流は cgl の A/blend でのみ（他力学は流れ場を持たない）
    if (dynamics === "cgl" && mode !== "B") engine.advectMap();
  };
  const loop = (): void => {
    const now = performance.now();
    const elapsed = now - last;
    last = now;
    if (running) {
      acc += Math.min(elapsed, 250); // タブ切替後の巨大 Δt はクランプ
      let iters = 0;
      while (acc >= LOGICAL_MS && iters < MAX_CATCHUP) {
        advance();
        acc -= LOGICAL_MS;
        iters++;
      }
      if (iters === MAX_CATCHUP) acc = 0; // 追いつけない時は積み残しを捨てる
    }
    engine.setState(params, modeNum(mode), blend, phaseRef);
    engine.render(gpu.context);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

void main();
