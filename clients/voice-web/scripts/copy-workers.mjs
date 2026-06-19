// Copy the opus-recorder worker scripts (and the decoder's sibling .wasm) into
// public/assets so they're served verbatim at /assets/*. opus-recorder's workers
// load their wasm by relative URL, which Vite's bundler mangles — serving them as
// static public files is the robust pattern (this is how the upstream Moshi
// client ships the decoder worker too). Runs from `predev`/`prebuild`.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const src = join(root, "node_modules", "opus-recorder", "dist");
const dst = join(root, "public", "assets");

mkdirSync(dst, { recursive: true });

const files = [
  "encoderWorker.min.js",
  "decoderWorker.min.js",
  "decoderWorker.min.wasm",
  "waveWorker.min.js",
];

let copied = 0;
for (const f of files) {
  const from = join(src, f);
  if (existsSync(from)) {
    copyFileSync(from, join(dst, f));
    copied++;
  } else {
    console.warn(`[copy-workers] missing ${f} (skipped)`);
  }
}
console.log(`[copy-workers] copied ${copied} worker file(s) to public/assets`);
