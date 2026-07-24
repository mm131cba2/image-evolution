// エントリ: WebGPU 初期化 → CGL エンジン → 種投入 → RAF ループ。
// フェーズ1 の最小形（モード B 表示）。まず既定の種で螺旋波を見せ、画像を選べば
// その輝度を位相に種付けして時間発展させる。

import { initWebGPU } from "./engine/gpu";
import { CGLEngine, type ModeNum } from "./engine/cglGPU";
import { buildCyclicLUT } from "./render/palette";
import { DEFAULT_CONFIG } from "./config";
import { imageToField, type Seed, type Field } from "./image/imageToField";
import { buildControls } from "./ui/controls";
import type { CGLParams } from "./engine/params";
import type { Mode } from "./config";

const L = DEFAULT_CONFIG.L; // 256

function modeNum(m: Mode): ModeNum {
  return m === "A" ? 0 : m === "B" ? 1 : 2;
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
  let lastFile: File | null = null;
  let running = true;
  // 共回転フレーム: CGL の一様位相回転 dθ/dt=−c を表示上で打ち消す（全画面の色ストロボ防止）。
  let coRotate = true;
  let phaseRef = 0;

  const apply = (): void => engine.setState(params, modeNum(mode), blend, phaseRef);

  const seedDefault = (): void => {
    const re = new Float32Array(L * L);
    const im = new Float32Array(L * L);
    for (let i = 0; i < L * L; i++) {
      const th = Math.random() * 2 * Math.PI;
      re[i] = Math.cos(th);
      im[i] = Math.sin(th);
    }
    engine.seed(re, im);
  };

  engine.seedOriginal(defaultOriginal());
  engine.resetMap();
  seedDefault();
  apply();

  const loadImage = (file: File): void => {
    void fieldFromFile(file, seedType).then((f) => {
      engine.seedOriginal(f.orig);
      engine.seed(f.psiRe, f.psiIm);
      engine.resetMap();
    });
  };

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
      engine.resetMap();
      if (lastFile) loadImage(lastFile);
      else seedDefault();
    },
    onFile: (file) => {
      lastFile = file;
      loadImage(file);
    },
    onSeedType: (s) => {
      seedType = s;
    },
    onCoRotate: (on) => {
      coRotate = on;
    },
    onTogglePause: () => {
      running = !running;
      return running;
    },
  });

  const STEPS_PER_FRAME = 6;
  const loop = (): void => {
    if (running) {
      engine.stepCGL(STEPS_PER_FRAME);
      // 一様回転ぶんだけ表示位相を戻す（dθ/dt=−c なので +c·Δt）
      if (coRotate) phaseRef += params.c * params.dt * STEPS_PER_FRAME;
      if (mode !== "B") engine.advectMap(); // A / blend のとき写真を流す
    }
    engine.setState(params, modeNum(mode), blend, phaseRef);
    engine.render(gpu.context);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

void main();
