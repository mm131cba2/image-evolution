# 画像時間発展アートツール

画像を初期状態に、複素 Ginzburg-Landau（CGL）で「時間発展」させる Web 作品。
サーバー不要・完全クライアントサイド（WebGPU）。

設計・判断・検証（なぜこの設計か）は別リポジトリの研究ノートにある
（`agent-notes/subtrees/works/pieces/image-evolution/` の spec.md / README.md / checks/）。

## 力学（2 モード＋ブレンド・共有 CGL コア）
- **A（flow）**: CGL 由来の流速で原本写真を流す＝写真無劣化・非往復・マーブリング（実装中）。
- **B（field）**: CGL の複素場 ψ を OKLCh 循環カラーマップで表示＝完全ループ・写真は種として褪せる（実装済）。
- A↔B は連続ブレンド（実装中）。

## 開発
```
npm install
npm run dev      # 開発サーバ（要 WebGPU: Chrome/Edge、Safari 26+、Firefox 最新）
npm test         # 純ロジックの単体テスト（Vitest）
npm run build    # dist/ に本番ビルド（base: './' ＝相対パス）
```
起動すると既定の種で螺旋波が動く。「画像を選ぶ」で画像の輝度を位相に種付けして発展。

## デプロイ（GitHub Pages: mm131cba2.github.io）
`base: './'`（相対パス）なのでサブパス配信に耐える。`npm run build` の `dist/` を Pages へ:
- 別リポジトリ `image-evolution` を作り `dist/` を公開 → `mm131cba2.github.io/image-evolution/`
- または `mm131cba2.github.io` リポジトリのサブフォルダに置く
- GitHub Actions で `npm run build` → `dist/` を `gh-pages` へ、が自動化として楽（後日）。

## 状態
純ロジック（パラメータ・画像→複素種・OKLCh LUT・設定/URL）＋ CGL コアは単体テスト済み（36 tests）。
GPU（WGSL・表示）はブラウザで手動検証。モード A の移流・ブレンド・UI は実装中。
