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
  seedQuat,
  seedScalar,
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
  const map: Record<Dynamics, DynNum> = {
    cgl: 0, grayscott: 1, lenia: 2, chroma: 3, quat: 4, scalar: 4, unified: 9,
    telegraph: 5, swifthohenberg: 6, fitzhugh: 7, cahnhilliard: 8,
  };
  return map[d];
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

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

// canvas を WebM 録画してダウンロード（依存ゼロ・MediaRecorder）。
async function recordCanvas(
  canvas: HTMLCanvasElement,
  seconds: number,
  onTick: (remaining: number) => void,
): Promise<void> {
  const stream = canvas.captureStream(30);
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
    ? "video/webm;codecs=vp9"
    : "video/webm";
  const chunks: Blob[] = [];
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 });
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  const stopped = new Promise<void>((res) => { rec.onstop = () => res(); });
  rec.start();
  for (let t = seconds; t > 0; t--) { onTick(t); await sleep(1000); }
  rec.stop();
  await stopped;
  const url = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `evolution-${Date.now()}.webm`;
  a.click();
  URL.revokeObjectURL(url);
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
  let anchorStrength = 0.6; // quat「元の色味を保つ」既定（0=自由発展・1=完全に元の重心）
  let quatM0: [number, number, number] = [0, 0, 0]; // 写真の色重心（seed 時に算出）
  let chromaYRate = 0.4; // chroma 輝度拡散率（0=Y固定=形保持・>0=形も溶ける）
  let gamma = 0.05; // telegraph 減衰（拡散↔波動の統一ツマミ・小=波動）
  let coupling = 1; // quat 代数結合 λ（1=四元数=色相回転・0=成分独立=褪色）
  let morph = 1; // lenia の diffusion↔Lenia 核 morph（1=パターン・0=拡散=均す）
  let pattern = 0; // unified の CGL↔SH パターンノブ p（0=四元数CGL・1=Swift-Hohenberg）
  let wave = 0; // unified の拡散↔波動ノブ a（0=拡散/緩和・→1=波動＝移流に慣性）

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
  // 直近に読み込んだ場（写真 or 既定）。力学/リセット切替は再デコードせず即座に再シード
  // する（切替時の非同期デコード待ちに古い状態を別力学が食う崩れを防ぐ）。
  let currentField: Field = defaultF;

  // 現在の力学に応じて engine に種を投入（写真か既定場の orig から状態を作る）。
  const seedFromField = (f: Field): void => {
    engine.seedOriginal(f.orig);
    if (dynamics === "cgl") {
      engine.seed(f.psiRe, f.psiIm);
    } else if (dynamics === "quat" || dynamics === "scalar" || dynamics === "unified") {
      // scalar＝quat の coupling=0 極限＝色の各成分が独立な実 Stuart-Landau 場（ℝ⊕ℝ⊕ℝ⊕ℝ・
      // 混ざらない・独立拡散で褪色）。unified＝統合形（四元数・CGL↔SH パターン）。いずれも quat の
      // vec4 場/表示/重心アンカーを共有し、結合 λ とパターン p だけ差し替える（checks/*）。
      const q4 = seedQuat(f.orig, L);
      // 写真の色重心 m0（純虚部 Im の平均）。表示の再センタの基準。
      let mx = 0, my = 0, mz = 0;
      const n = L * L;
      for (let i = 0; i < n; i++) { mx += q4[i * 4 + 1]; my += q4[i * 4 + 2]; mz += q4[i * 4 + 3]; }
      quatM0 = [mx / n, my / n, mz / n];
      engine.setQuatAnchor(anchorStrength, quatM0);
      engine.setCoupling(dynamics === "scalar" ? 0 : coupling); // scalar は結合ゼロ＝成分独立
      engine.setPattern(dynamics === "unified" ? pattern : 0);  // pattern は unified のみ
      engine.setWave(dynamics === "unified" ? wave : 0);        // wave も unified のみ
      if (dynamics === "unified") engine.resetVel();            // 波動の慣性を 0 から
      engine.seedQuat(q4);
    } else if (dynamics === "chroma") {
      engine.setChromaYRate(chromaYRate);
      engine.seedQuat(seedChroma(f.orig, L)); // vec4=(Y,Cb,Cr,0)
    } else {
      // vec2 スカラー/2成分力学。種は写真の輝度から（力学ごとに振幅）。
      const s =
        dynamics === "grayscott" ? seedGrayScott(f.orig, L)
        : dynamics === "lenia" ? seedLenia(f.orig, L)
        : dynamics === "telegraph" ? seedScalar(f.orig, L, 1.0)
        : dynamics === "swifthohenberg" ? seedScalar(f.orig, L, 0.2)
        : dynamics === "fitzhugh" ? seedScalar(f.orig, L, 2.0)
        : seedScalar(f.orig, L, 0.4); // cahnhilliard
      if (dynamics === "telegraph") engine.setGamma(gamma);
      if (dynamics === "lenia") engine.setMorph(morph);
      engine.seed(s.re, s.im);
    }
    engine.resetMap();
    phaseRef = 0; // t=0 に戻す（共回転オフセットもリセット）
  };
  const reseed = (): void => seedFromField(currentField);

  // 写真を読み込む（seedType/dynamics 変更で psi の作り直しが要る時だけ再デコード）。
  const loadFile = (file: File): void => {
    void fieldFromFile(file, seedType).then((f) => {
      currentField = f;
      seedFromField(f);
    });
  };

  engine.setDynamics(dynNum(dynamics));
  reseed();
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
      reseed(); // 保持中の場から即座に（決定的）
    },
    onFile: (file) => {
      lastFile = file;
      loadFile(file);
    },
    onSeedType: (s) => {
      seedType = s;
      if (lastFile) loadFile(lastFile); // psi の作り直しに再デコードが要る（cgl のみ効く）
      else reseed();
    },
    onDynamics: (d) => {
      dynamics = d;
      engine.setDynamics(dynNum(d));
      reseed(); // 保持中の場から力学に応じて即再シード（非同期デコード待ちを挟まない）
    },
    onCoRotate: (on) => {
      coRotate = on;
    },
    onAnchor: (v) => {
      anchorStrength = v;
      engine.setQuatAnchor(anchorStrength, quatM0); // quat の「元の色味を保つ」強度
    },
    onYRate: (v) => {
      chromaYRate = v;
      engine.setChromaYRate(v); // chroma の輝度拡散率
    },
    onGamma: (v) => {
      gamma = v;
      engine.setGamma(v); // telegraph の拡散↔波動
    },
    onCoupling: (v) => {
      coupling = v;
      engine.setCoupling(v); // quat の scalar↔quat 代数結合
    },
    onMorph: (v) => {
      morph = v;
      engine.setMorph(v); // lenia の diffusion↔Lenia 核 morph
    },
    onPattern: (v) => {
      pattern = v;
      engine.setPattern(v); // unified の CGL↔SH パターンノブ
    },
    onWave: (v) => {
      wave = v;
      engine.setWave(v); // unified の拡散↔波動ノブ
    },
    onRecord: (seconds, onTick) => recordCanvas(canvas, seconds, onTick),
    onTogglePause: () => {
      running = !running;
      return running;
    },
  });

  // フレームレート非依存: 実時間を貯めて 1/60s ぶんの論理フレームだけ進める。
  // 60Hz では毎フレーム 1 回（従来と同一）・120Hz では 1 回おき・30Hz では 2 回進み、
  // 表示装置に依らず同じ速さで時間発展する（ループ書き出しの再現性にも効く）。
  const LOGICAL_MS = 1000 / 60;
  const MAX_CATCHUP = 4; // 背景タブ復帰などの暴走を防ぐ上限
  // 力学ごとの 1 論理フレーム当たりステップ数（cgl=標準／lenia=重いので少なめ／
  // chroma=拡散が遅く見えるので多め／grayscott=見やすくやや少なめ）。
  const stepsFor = (d: Dynamics): number =>
    d === "lenia" ? 2 : d === "chroma" ? 20 : d === "grayscott" ? 3
    : d === "fitzhugh" ? 3 : d === "cahnhilliard" ? 8 : 6; // telegraph/swift/quat/cgl=6
  let acc = 0;
  let last = performance.now();
  const advance = (): void => {
    const steps = stepsFor(dynamics);
    engine.stepCGL(steps);
    if (coRotate) phaseRef += params.c * params.dt * steps; // +c·Δt で一様回転を相殺
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
    if (dynamics === "quat" || dynamics === "scalar" || dynamics === "unified") engine.computeQuatMean(); // 重心を更新してから描画
    engine.render(gpu.context);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

void main();
