// エントリ: WebGPU 初期化 → CGL エンジン → 種投入 → RAF ループ。
// フェーズ1 の最小形（モード B 表示）。まず既定の種で螺旋波を見せ、画像を選べば
// その輝度を位相に種付けして時間発展させる。

import { initWebGPU } from "./engine/gpu";
import { CGLEngine } from "./engine/cglGPU";
import { buildCyclicLUT } from "./render/palette";
import { DEFAULT_CONFIG } from "./config";
import { imageToField } from "./image/imageToField";

const L = DEFAULT_CONFIG.L; // 256

function overlay(msg: string): void {
  const d = document.createElement("div");
  d.style.cssText =
    "position:fixed;inset:0;display:flex;align-items:center;justify-content:center;" +
    "color:#ccc;font-family:sans-serif;font-size:16px;padding:2rem;text-align:center;";
  d.textContent = msg;
  document.body.appendChild(d);
}

// 既定の種: ランダム位相（|ψ|=1）。CGL が螺旋波へ組織化する。
function defaultSeed(): { re: Float32Array; im: Float32Array } {
  const re = new Float32Array(L * L);
  const im = new Float32Array(L * L);
  for (let i = 0; i < L * L; i++) {
    const th = Math.random() * 2 * Math.PI;
    re[i] = Math.cos(th);
    im[i] = Math.sin(th);
  }
  return { re, im };
}

// 画像ファイル → L×L の場（面積平均は imageToField 側）。
async function fieldFromFile(file: File): Promise<{ re: Float32Array; im: Float32Array }> {
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
  const f = imageToField(id, L, DEFAULT_CONFIG.seed);
  return { re: f.psiRe, im: f.psiIm };
}

function buildControls(onFile: (f: File) => void, onToggle: () => boolean): void {
  const bar = document.createElement("div");
  bar.style.cssText =
    "position:fixed;top:10px;left:10px;display:flex;gap:8px;align-items:center;" +
    "font-family:sans-serif;font-size:13px;color:#ddd;background:rgba(0,0,0,.45);" +
    "padding:6px 10px;border-radius:8px;";

  const label = document.createElement("label");
  label.textContent = "画像を選ぶ";
  label.style.cssText = "cursor:pointer;text-decoration:underline;";
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.style.display = "none";
  input.addEventListener("change", () => {
    const f = input.files?.[0];
    if (f) onFile(f);
  });
  label.appendChild(input);

  const pause = document.createElement("button");
  pause.textContent = "⏸ 一時停止";
  pause.style.cssText = "cursor:pointer;";
  pause.addEventListener("click", () => {
    pause.textContent = onToggle() ? "⏸ 一時停止" : "▶ 再生";
  });

  bar.appendChild(label);
  bar.appendChild(pause);
  document.body.appendChild(bar);
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
  engine.setParams(DEFAULT_CONFIG.params);
  const s0 = defaultSeed();
  engine.seed(s0.re, s0.im);

  let running = true;
  buildControls(
    (file) => {
      void fieldFromFile(file).then((f) => engine.seed(f.re, f.im));
    },
    () => {
      running = !running;
      return running;
    },
  );

  const STEPS_PER_FRAME = 6;
  const loop = (): void => {
    if (running) engine.step(STEPS_PER_FRAME);
    engine.render(gpu.context);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

void main();
