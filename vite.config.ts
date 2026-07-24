/// <reference types="vitest/config" />
import { defineConfig } from "vite";

// base: "./" は GitHub Pages で相対パスにするため（サブパス配信に耐える）。
export default defineConfig({
  base: "./",
  test: {
    environment: "node", // 純ロジックは node。DOM が要るテストはファイル先頭で jsdom を指定する。
  },
});
