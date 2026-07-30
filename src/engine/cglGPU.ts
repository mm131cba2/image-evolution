// WebGPU の CGL エンジン（モード A/B/blend）。
// ψ を storage buffer(vec2) のピンポンで更新し、逆写像 map で原本を流す（A）／OKLCh で描く（B）。
// 数値は CPU 参照 src/engine/cglStep.ts と同一（WGSL は shaders.ts）。

import {
  CGL_STEP_WGSL,
  GRAYSCOTT_WGSL,
  LENIA_WGSL,
  CHROMA_WGSL,
  QUAT_STEP_WGSL,
  UNIFIED_QUAT_STEP_WGSL,
  QUAT_MEAN_WGSL,
  TELEGRAPH_WGSL,
  SWIFT_WGSL,
  FHN_WGSL,
  CAHN_WGSL,
  ADVECT_WGSL,
  ADVECT_MC2_WGSL,
  DISPLAY_WGSL,
} from "./shaders";
import { linearToSrgb } from "../color";
import type { CGLParams } from "./params";

const PARAMS_BYTES = 96;
export type ModeNum = 0 | 1 | 2; // 0=A(flow) 1=B(field) 2=blend
// 0=cgl 1=grayscott 2=lenia 3=chroma 4=quat 5=telegraph 6=swift 7=fhn 8=cahn 9=unified(quat)
export type DynNum = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

export class CGLEngine {
  private cur = 0; // psi parity（最新）
  private mapCur = 0;
  private buffers: [GPUBuffer, GPUBuffer]; // ψ
  private maps: [GPUBuffer, GPUBuffer]; // 逆写像(ピクセル座標 vec2)
  private mapTmp: GPUBuffer; // MacCormack 第1パス φ1
  private paramsBuf: GPUBuffer;
  private lutTex: GPUTexture;
  private origTex: GPUTexture;
  private sampler: GPUSampler;
  private origSampler: GPUSampler;
  private stepPipelines: Record<number, GPUComputePipeline>; // vec2 力学（dynamics 番号→pipeline）
  private qbuf: [GPUBuffer, GPUBuffer]; // vec4 場（chroma=YCbCr / quat）のピンポン
  private chromaPipeline: GPUComputePipeline; // vec4 色拡散
  private quatPipeline: GPUComputePipeline;
  private unifiedPipeline: GPUComputePipeline; // 統合形（四元数版）・速度チャンネル付き専用レイアウト
  private unifiedBG: [GPUBindGroup, GPUBindGroup];
  private qvel: GPUBuffer; // 統合形の移流速度チャンネル（telegraph・vec4）
  private quatBG: [GPUBindGroup, GPUBindGroup];
  private meanBuf: GPUBuffer; // 共回転後 Im 重心（vec4）
  private quatMeanPipeline: GPUComputePipeline;
  private quatMeanBG: [GPUBindGroup, GPUBindGroup];
  private advectPipeline: GPUComputePipeline;
  private advectPipeline2: GPUComputePipeline;
  private renderPipeline: GPURenderPipeline;
  private computeBG: [GPUBindGroup, GPUBindGroup];
  private paramsHost = new ArrayBuffer(PARAMS_BYTES);
  private dynamics: DynNum = 0;
  private anchor = 0; // 重心アンカー強度（quat・0=自由）
  private m0: [number, number, number] = [0, 0, 0]; // 写真の Im 重心
  private yrate = 0; // chroma の輝度拡散率（0=Y固定=形保持・>0=Yも溶ける）
  private gamma = 0.05; // telegraph の減衰（0→波動・大→拡散）
  private coupling = 1; // quat の代数結合 λ（1=四元数=色相回転・0=成分独立=褪色）
  private morph = 1; // lenia の diffusion↔Lenia 核 morph（1=Lenia パターン・0=拡散=均す）
  private pattern = 0; // 統合形の CGL↔SH パターンノブ（0=四元数CGL・1=Swift-Hohenberg）
  private wave = 0; // 統合形の拡散↔波動ノブ a（0=拡散/緩和=1階・→1=波動=移流に慣性）

  constructor(
    private device: GPUDevice,
    format: GPUTextureFormat,
    private L: number,
    lut: Uint8Array<ArrayBuffer>,
  ) {
    const n = L * L;
    const mkStorage = (bytes: number) =>
      device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.buffers = [mkStorage(n * 2 * 4), mkStorage(n * 2 * 4)];
    this.maps = [mkStorage(n * 2 * 4), mkStorage(n * 2 * 4)];
    this.mapTmp = mkStorage(n * 2 * 4);
    this.qbuf = [mkStorage(n * 4 * 4), mkStorage(n * 4 * 4)]; // vec4
    this.qvel = mkStorage(n * 4 * 4); // 統合形の移流速度（telegraph・vec4・ピンポン不要）
    this.meanBuf = mkStorage(4 * 4); // vec4（Im 重心）

    this.paramsBuf = device.createBuffer({
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.lutTex = device.createTexture({
      size: [256, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: this.lutTex }, lut, { bytesPerRow: 256 * 4 }, [256, 1]);

    this.origTex = device.createTexture({
      size: [L, L],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "repeat" });
    this.origSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    const advMod = device.createShaderModule({ code: ADVECT_WGSL });
    const adv2Mod = device.createShaderModule({ code: ADVECT_MC2_WGSL });
    const dispMod = device.createShaderModule({ code: DISPLAY_WGSL });

    // 全力学ステップは同一バインディング（in 読取/out 書込/params）＝共有レイアウトで
    // 1 組のバインドグループを使い回す（auto だとパイプライン毎に非互換になる）。
    const stepBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const stepPL = device.createPipelineLayout({ bindGroupLayouts: [stepBGL] });
    const mkStep = (code: string) =>
      device.createComputePipeline({
        layout: stepPL,
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      });
    // vec2 力学（dynamics 番号→pipeline・computeBG を共有）。3/4 は vec4 で別扱い。
    this.stepPipelines = {
      0: mkStep(CGL_STEP_WGSL),
      1: mkStep(GRAYSCOTT_WGSL),
      2: mkStep(LENIA_WGSL),
      5: mkStep(TELEGRAPH_WGSL),
      6: mkStep(SWIFT_WGSL),
      7: mkStep(FHN_WGSL),
      8: mkStep(CAHN_WGSL),
    };
    // vec4 場（chroma=YCbCr / quat）も同じ step レイアウト（storage in/out + uniform）を共有。
    this.chromaPipeline = mkStep(CHROMA_WGSL);
    this.quatPipeline = mkStep(QUAT_STEP_WGSL);
    this.quatBG = [
      device.createBindGroup({
        layout: stepBGL,
        entries: [
          { binding: 0, resource: { buffer: this.qbuf[0] } },
          { binding: 1, resource: { buffer: this.qbuf[1] } },
          { binding: 2, resource: { buffer: this.paramsBuf } },
        ],
      }),
      device.createBindGroup({
        layout: stepBGL,
        entries: [
          { binding: 0, resource: { buffer: this.qbuf[1] } },
          { binding: 1, resource: { buffer: this.qbuf[0] } },
          { binding: 2, resource: { buffer: this.paramsBuf } },
        ],
      }),
    ];
    // 統合形は速度チャンネル(binding 3・read_write storage)を持つ専用 4 バインドレイアウト。
    const unifiedBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.unifiedPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [unifiedBGL] }),
      compute: { module: device.createShaderModule({ code: UNIFIED_QUAT_STEP_WGSL }), entryPoint: "main" },
    });
    const mkUnified = (inB: GPUBuffer, outB: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        layout: unifiedBGL,
        entries: [
          { binding: 0, resource: { buffer: inB } },
          { binding: 1, resource: { buffer: outB } },
          { binding: 2, resource: { buffer: this.paramsBuf } },
          { binding: 3, resource: { buffer: this.qvel } },
        ],
      });
    this.unifiedBG = [mkUnified(this.qbuf[0], this.qbuf[1]), mkUnified(this.qbuf[1], this.qbuf[0])];

    // 重心縮約（qbuf[cur] → meanBuf）。1 ワークグループ・auto レイアウト。
    this.quatMeanPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: QUAT_MEAN_WGSL }), entryPoint: "main" },
    });
    const mml = this.quatMeanPipeline.getBindGroupLayout(0);
    const mkMean = (qb: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        layout: mml,
        entries: [
          { binding: 0, resource: { buffer: qb } },
          { binding: 1, resource: { buffer: this.paramsBuf } },
          { binding: 2, resource: { buffer: this.meanBuf } },
        ],
      });
    this.quatMeanBG = [mkMean(this.qbuf[0]), mkMean(this.qbuf[1])];
    this.advectPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: advMod, entryPoint: "main" },
    });
    this.advectPipeline2 = device.createComputePipeline({
      layout: "auto",
      compute: { module: adv2Mod, entryPoint: "main" },
    });
    this.renderPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: dispMod, entryPoint: "vs" },
      fragment: { module: dispMod, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });

    const mkCompute = (inB: GPUBuffer, outB: GPUBuffer) =>
      device.createBindGroup({
        layout: stepBGL,
        entries: [
          { binding: 0, resource: { buffer: inB } },
          { binding: 1, resource: { buffer: outB } },
          { binding: 2, resource: { buffer: this.paramsBuf } },
        ],
      });
    this.computeBG = [mkCompute(this.buffers[0], this.buffers[1]), mkCompute(this.buffers[1], this.buffers[0])];

    this.resetMap();
  }

  setDynamics(d: DynNum): void {
    this.dynamics = d;
  }

  // 重心アンカー（quat 全色発展の「元の色味を保つ」強度と写真の Im 重心）。
  setQuatAnchor(strength: number, m0: [number, number, number]): void {
    this.anchor = strength;
    this.m0 = m0;
  }

  // chroma の輝度拡散率（0=Y 固定＝形保持・>0=Y も拡散して溶ける）。
  setChromaYRate(r: number): void {
    this.yrate = r;
  }

  // telegraph の減衰 γ（0→波動・大→拡散）。拡散↔波動の統一ツマミ。
  setGamma(g: number): void {
    this.gamma = g;
  }

  // quat の代数結合 λ（1=四元数=色相回転内蔵・0=成分独立=直和ℝ⁴で褪色）。scalar↔quat ノブ。
  setCoupling(l: number): void {
    this.coupling = l;
  }

  // lenia の diffusion↔Lenia 核 morph（1=リング核+ベル=パターン・0=局所核+恒等=拡散で均す）。
  setMorph(m: number): void {
    this.morph = m;
  }

  // 統合形（四元数）の CGL↔SH パターンノブ p（0=四元数CGL＝k=0 リミットサイクル・1=Swift-Hohenberg＝有限 k 模様）。
  setPattern(p: number): void {
    this.pattern = p;
  }

  // 統合形（四元数）の拡散↔波動ノブ a（0=拡散/緩和=1階・→1=波動＝移流に慣性・双曲型反応拡散）。
  setWave(a: number): void {
    this.wave = a;
  }

  // 統合形の移流速度チャンネルを 0 に（再シード時＝波動の慣性をリセットして決定的に開始）。
  resetVel(): void {
    this.device.queue.writeBuffer(this.qvel, 0, new Float32Array(this.L * this.L * 4));
  }

  setState(p: CGLParams, mode: ModeNum, blend: number, phaseRef = 0, ampRef = 1.0): void {
    const dv = new DataView(this.paramsHost);
    dv.setUint32(0, this.L, true);
    dv.setFloat32(4, p.b, true);
    dv.setFloat32(8, p.c, true);
    dv.setFloat32(12, p.D, true);
    dv.setFloat32(16, p.dt, true);
    dv.setFloat32(20, ampRef, true);
    dv.setFloat32(24, p.speed, true);
    dv.setUint32(28, mode, true);
    dv.setFloat32(32, blend, true);
    dv.setFloat32(36, phaseRef, true);
    dv.setUint32(40, this.dynamics, true);
    dv.setFloat32(44, this.anchor, true);
    dv.setFloat32(48, this.m0[0], true);
    dv.setFloat32(52, this.m0[1], true);
    dv.setFloat32(56, this.m0[2], true);
    dv.setFloat32(60, this.yrate, true);
    dv.setFloat32(64, this.gamma, true);
    dv.setFloat32(68, this.coupling, true);
    dv.setFloat32(72, this.morph, true);
    dv.setFloat32(76, this.pattern, true);
    dv.setFloat32(80, this.wave, true);
    this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsHost);
  }

  seed(re: Float32Array, im: Float32Array): void {
    const n = this.L * this.L;
    const inter = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      inter[i * 2] = re[i];
      inter[i * 2 + 1] = im[i];
    }
    this.device.queue.writeBuffer(this.buffers[this.cur], 0, inter);
  }

  // 四元数場に種を投入（q4 = n*4 の interleave (w,x,y,z)）。
  seedQuat(q4: Float32Array<ArrayBuffer>): void {
    this.device.queue.writeBuffer(this.qbuf[this.cur], 0, q4);
  }

  // 四元数場の共回転後 Im 重心を meanBuf に縮約（表示の再センタ用・毎フレーム quat 時）。
  computeQuatMean(): void {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.quatMeanPipeline);
    pass.setBindGroup(0, this.quatMeanBG[this.cur]);
    pass.dispatchWorkgroups(1);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  // 原本テクスチャを投入（orig は線形光 RGBA・L*L*4）。表示は sRGB で行うので変換して格納。
  seedOriginal(orig: Float32Array): void {
    const n = this.L * this.L;
    const bytes = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) {
      bytes[i * 4] = Math.round(linearToSrgb(orig[i * 4]) * 255);
      bytes[i * 4 + 1] = Math.round(linearToSrgb(orig[i * 4 + 1]) * 255);
      bytes[i * 4 + 2] = Math.round(linearToSrgb(orig[i * 4 + 2]) * 255);
      bytes[i * 4 + 3] = 255;
    }
    this.device.queue.writeTexture({ texture: this.origTex }, bytes, { bytesPerRow: this.L * 4 }, [this.L, this.L]);
  }

  // 逆写像を恒等（各出力画素は自分の位置＝原本をそのまま）に初期化。
  resetMap(): void {
    const L = this.L;
    const ident = new Float32Array(L * L * 2);
    for (let y = 0; y < L; y++) {
      for (let x = 0; x < L; x++) {
        const i = (y * L + x) * 2;
        ident[i] = x;
        ident[i + 1] = y;
      }
    }
    this.device.queue.writeBuffer(this.maps[0], 0, ident);
    this.device.queue.writeBuffer(this.maps[1], 0, ident);
    this.mapCur = 0;
  }

  // 現在の力学を times ステップ進める（ピンポン）。力学は setDynamics で選択。
  // dynamics 3(chroma)/4(quat)/9(unified) は vec4 場(qbuf)を、それ以外は vec2 場(buffers)を進める。
  stepCGL(times: number): void {
    const isVec4 = this.dynamics === 3 || this.dynamics === 4 || this.dynamics === 9;
    const pipe = this.dynamics === 4 ? this.quatPipeline
      : this.dynamics === 9 ? this.unifiedPipeline
      : this.dynamics === 3 ? this.chromaPipeline
      : this.stepPipelines[this.dynamics];
    const bgs = this.dynamics === 9 ? this.unifiedBG : isVec4 ? this.quatBG : this.computeBG;
    const enc = this.device.createCommandEncoder();
    const wg = Math.ceil(this.L / 8);
    for (let i = 0; i < times; i++) {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bgs[this.cur]);
      pass.dispatchWorkgroups(wg, wg);
      pass.end();
      this.cur ^= 1;
    }
    this.device.queue.submit([enc.finish()]);
  }

  // 逆写像を ψ 由来の速度で移流（MacCormack 2 パス・写真を鮮鋭に保つ）。
  advectMap(): void {
    const psi = this.buffers[this.cur];
    const wg = Math.ceil(this.L / 8);
    const enc = this.device.createCommandEncoder();

    // パス1: mapCur → mapTmp（後退 semi-Lagrangian）
    const bg1 = this.device.createBindGroup({
      layout: this.advectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.maps[this.mapCur] } },
        { binding: 1, resource: { buffer: this.mapTmp } },
        { binding: 2, resource: { buffer: psi } },
        { binding: 3, resource: { buffer: this.paramsBuf } },
      ],
    });
    const p1 = enc.beginComputePass();
    p1.setPipeline(this.advectPipeline);
    p1.setBindGroup(0, bg1);
    p1.dispatchWorkgroups(wg, wg);
    p1.end();

    // パス2: 誤差補正 → maps[mapCur^1]
    const bg2 = this.device.createBindGroup({
      layout: this.advectPipeline2.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.mapTmp } },
        { binding: 1, resource: { buffer: this.maps[this.mapCur] } },
        { binding: 2, resource: { buffer: this.maps[this.mapCur ^ 1] } },
        { binding: 3, resource: { buffer: psi } },
        { binding: 4, resource: { buffer: this.paramsBuf } },
      ],
    });
    const p2 = enc.beginComputePass();
    p2.setPipeline(this.advectPipeline2);
    p2.setBindGroup(0, bg2);
    p2.dispatchWorkgroups(wg, wg);
    p2.end();

    this.device.queue.submit([enc.finish()]);
    this.mapCur ^= 1;
  }

  render(context: GPUCanvasContext): void {
    const bg = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers[this.cur] } },
        { binding: 1, resource: { buffer: this.paramsBuf } },
        { binding: 2, resource: this.lutTex.createView() },
        { binding: 3, resource: this.sampler },
        { binding: 4, resource: { buffer: this.maps[this.mapCur] } },
        { binding: 5, resource: this.origTex.createView() },
        { binding: 6, resource: this.origSampler },
        { binding: 7, resource: { buffer: this.qbuf[this.cur] } },
        { binding: 8, resource: { buffer: this.meanBuf } },
      ],
    });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0.07, g: 0.07, b: 0.07, a: 1 },
        },
      ],
    });
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
