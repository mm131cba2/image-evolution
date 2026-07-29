// シームレスループ書き出しの中核（純ロジック・GPU 非依存）。
//
// 再生中に各フレームを低次元の特徴ベクトルへ縮約して溜め、「start から length
// フレーム後に状態がどれだけ近く戻るか」を最小化する (start, length) を探す
// ＝Video Textures [Schödl2000] の継ぎ目探索。
//   モード B（CGL リミットサイクル）: length≈周期で継ぎ目≈0＝完全ループ。
//   モード A（非往復マーブリング）: 完全には戻らないので最も目立たない近似ループを選ぶ。
// 継ぎ目探索は特徴列だけで完結する純ロジックなので Vitest で検証できる（GPU 不要）。

// L×L・comps 成分/セルの場（例: ψ は comps=2 の re/im 交互、quat は comps=4）を
// K×K ブロック平均で特徴ベクトルへ縮約（長さ comps*K*K）。継ぎ目探索を安くし
// 微小ノイズに頑健にする。全画素総当たり（256²）は書き出しには重すぎるため。
export function downsampleField(
  field: Float32Array,
  L: number,
  comps: number,
  K: number,
): Float32Array {
  const feat = new Float32Array(comps * K * K);
  const counts = new Float32Array(K * K);
  for (let y = 0; y < L; y++) {
    const by = Math.min(K - 1, Math.floor((y * K) / L));
    for (let x = 0; x < L; x++) {
      const bx = Math.min(K - 1, Math.floor((x * K) / L));
      const b = by * K + bx;
      const src = (y * L + x) * comps;
      for (let k = 0; k < comps; k++) feat[b * comps + k] += field[src + k];
      counts[b] += 1;
    }
  }
  for (let b = 0; b < K * K; b++) {
    const n = counts[b] || 1;
    for (let k = 0; k < comps; k++) feat[b * comps + k] /= n;
  }
  return feat;
}

// 2 つの特徴ベクトルの L2 距離（継ぎ目のなめらかさ＝小さいほど良い）。
export function l2dist(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

export interface LoopResult {
  start: number; // ループ開始フレーム index
  length: number; // ループ長＝周期の推定（フレーム数）
  cost: number; // 継ぎ目コスト（特徴 L2 距離の窓和・小さいほど滑らか）
}

export interface LoopOptions {
  minLength: number; // 最小ループ長（フレーム・短すぎるループを弾く）
  maxLength: number; // 最大ループ長
  window?: number; // 継ぎ目で照合する連続フレーム数（速度＝時間微分も揃える・既定 2）
}

// 溜めた特徴列から最良のシームレスループ (start, length) を探す。
// 単フレームの一致だけでなく window フレームを照合するので、位置だけでなく
// 「動きの向き（速度）」も揃った継ぎ目を選ぶ（Schödl の dynamics preservation）。
// length を昇順に走査し strict < で更新するため、同コストなら最短周期を選ぶ
// （リミットサイクルの 1 周期を 2 周期と誤検出しない）。
export function bestLoop(features: Float32Array[], opts: LoopOptions): LoopResult {
  const N = features.length;
  const window = Math.max(1, Math.floor(opts.window ?? 2));
  const minLen = Math.max(1, Math.floor(opts.minLength));
  const maxLen = Math.min(Math.floor(opts.maxLength), N - window);
  let best: LoopResult = { start: 0, length: minLen, cost: Infinity };
  for (let lag = minLen; lag <= maxLen; lag++) {
    for (let s = 0; s + lag + window - 1 < N; s++) {
      let cost = 0;
      for (let w = 0; w < window; w++) {
        cost += l2dist(features[s + w], features[s + lag + w]);
      }
      if (cost < best.cost) best = { start: s, length: lag, cost };
    }
  }
  return best;
}

// 継ぎ目を隠すクロスフェード重み（長さ fade）。ループ末尾の fade フレームを
// 先頭 fade フレームへ線形に混ぜる: out = mix(main, wrapped, w[i])。
// w は 0→1 の単調増加（端で 0/1 に達しないよう内側にオフセット）。
export function crossfadeWeights(fade: number): Float32Array {
  const n = Math.max(0, Math.floor(fade));
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = (i + 1) / (n + 1);
  return w;
}
