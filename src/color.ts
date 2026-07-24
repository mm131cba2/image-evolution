// 色変換の共通実装（image / render で共有）。
// 規律: 力学の計算は「線形光（混色系）」で行い、知覚的な調整は表示側（パレット）で行う。
// 参照: 設計ノート checks/algo.py（ガンマ）・checks/color.py / checks/complex-color.py（OKLab）。

// sRGB [0,1] ↔ 線形光 [0,1]（IEC 61966-2-1）。
export function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export function linearToSrgb(c: number): number {
  const x = Math.min(1, Math.max(0, c));
  return x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055;
}

// 線形光の輝度（Rec.709）。ガンマ済み sRGB 値には使わない（線形光に対して定義）。
export function luminance709(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// --- OKLab（Ottosson 2020）: 表示の知覚均等パレット用 -------------------------
// 線形 sRGB ↔ OKLab。係数は原典から。

export function linearToOklab(r: number, g: number, b: number): [number, number, number] {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

export function oklabToLinear(L: number, a: number, b: number): [number, number, number] {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}

// OKLCh（極形式）→ 線形 sRGB。h はラジアン。
export function oklchToLinear(L: number, C: number, h: number): [number, number, number] {
  return oklabToLinear(L, C * Math.cos(h), C * Math.sin(h));
}

export function inGamut([r, g, b]: [number, number, number], eps = 1e-3): boolean {
  return r >= -eps && r <= 1 + eps && g >= -eps && g <= 1 + eps && b >= -eps && b <= 1 + eps;
}
