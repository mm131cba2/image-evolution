// WebGPU の CGL エンジン（フェーズ1・モード B 表示まで）。
// ψ を storage buffer(vec2) のピンポンで更新し、OKLCh 循環 LUT で描く。
// 数値は CPU 参照 src/engine/cglStep.ts と同一（WGSL は shaders.ts で 1:1 移植）。

import { CGL_STEP_WGSL, DISPLAY_WGSL } from "./shaders";
import type { CGLParams } from "./params";

const PARAMS_BYTES = 32;

export class CGLEngine {
  private cur = 0;
  private buffers: [GPUBuffer, GPUBuffer];
  private paramsBuf: GPUBuffer;
  private lutTex: GPUTexture;
  private sampler: GPUSampler;
  private computePipeline: GPUComputePipeline;
  private renderPipeline: GPURenderPipeline;
  private computeBG: [GPUBindGroup, GPUBindGroup];
  private renderBG: [GPUBindGroup, GPUBindGroup];
  private paramsHost = new ArrayBuffer(PARAMS_BYTES);

  constructor(
    private device: GPUDevice,
    format: GPUTextureFormat,
    private L: number,
    lut: Uint8Array<ArrayBuffer>,
  ) {
    const n = L * L;
    const mk = () =>
      device.createBuffer({
        size: n * 2 * 4, // vec2<f32>
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    this.buffers = [mk(), mk()];

    this.paramsBuf = device.createBuffer({
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // OKLCh 循環 LUT（256×1・rgba8unorm・中身は sRGB バイト）。
    this.lutTex = device.createTexture({
      size: [256, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: this.lutTex }, lut, { bytesPerRow: 256 * 4 }, [256, 1]);
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat", // 色相は循環
    });

    const stepMod = device.createShaderModule({ code: CGL_STEP_WGSL });
    const dispMod = device.createShaderModule({ code: DISPLAY_WGSL });

    this.computePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: stepMod, entryPoint: "main" },
    });
    this.renderPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: dispMod, entryPoint: "vs" },
      fragment: { module: dispMod, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });

    const cl = this.computePipeline.getBindGroupLayout(0);
    const makeCompute = (inB: GPUBuffer, outB: GPUBuffer) =>
      device.createBindGroup({
        layout: cl,
        entries: [
          { binding: 0, resource: { buffer: inB } },
          { binding: 1, resource: { buffer: outB } },
          { binding: 2, resource: { buffer: this.paramsBuf } },
        ],
      });
    this.computeBG = [
      makeCompute(this.buffers[0], this.buffers[1]),
      makeCompute(this.buffers[1], this.buffers[0]),
    ];

    const rl = this.renderPipeline.getBindGroupLayout(0);
    const makeRender = (b: GPUBuffer) =>
      device.createBindGroup({
        layout: rl,
        entries: [
          { binding: 0, resource: { buffer: b } },
          { binding: 1, resource: { buffer: this.paramsBuf } },
          { binding: 2, resource: this.lutTex.createView() },
          { binding: 3, resource: this.sampler },
        ],
      });
    this.renderBG = [makeRender(this.buffers[0]), makeRender(this.buffers[1])];
  }

  setParams(p: CGLParams, ampRef = 1.0): void {
    const dv = new DataView(this.paramsHost);
    dv.setUint32(0, this.L, true);
    dv.setFloat32(4, p.b, true);
    dv.setFloat32(8, p.c, true);
    dv.setFloat32(12, p.D, true);
    dv.setFloat32(16, p.dt, true);
    dv.setFloat32(20, ampRef, true);
    this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsHost);
  }

  // 複素種を投入（re, im は L*L）。現在のバッファに書く。
  seed(re: Float32Array, im: Float32Array): void {
    const n = this.L * this.L;
    const inter = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      inter[i * 2] = re[i];
      inter[i * 2 + 1] = im[i];
    }
    this.device.queue.writeBuffer(this.buffers[this.cur], 0, inter);
  }

  // times ステップ進める（1 フレームで複数ステップ回して見やすい速度に）。
  step(times: number): void {
    const enc = this.device.createCommandEncoder();
    const wg = Math.ceil(this.L / 8);
    for (let i = 0; i < times; i++) {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, this.computeBG[this.cur]);
      pass.dispatchWorkgroups(wg, wg);
      pass.end();
      this.cur ^= 1;
    }
    this.device.queue.submit([enc.finish()]);
  }

  render(context: GPUCanvasContext): void {
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
    pass.setBindGroup(0, this.renderBG[this.cur]); // cur は最新結果のバッファ
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
