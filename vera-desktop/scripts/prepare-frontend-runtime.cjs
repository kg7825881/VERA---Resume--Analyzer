const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..", "..");
const standalone = path.join(root, "vera-frontend", ".next", "standalone");
const staticAssets = path.join(root, "vera-frontend", ".next", "static");
const target = path.join(root, "vera-desktop", "frontend-runtime");
const dependenciesTarget = path.join(root, "vera-desktop", "frontend-deps");

if (!fs.existsSync(standalone)) {
  throw new Error("Missing standalone frontend build. Run `npm run build` in vera-frontend first.");
}

fs.rmSync(target, { recursive: true, force: true });
fs.rmSync(dependenciesTarget, { recursive: true, force: true });
fs.cpSync(standalone, target, { recursive: true });
fs.mkdirSync(path.join(target, ".next"), { recursive: true });
fs.cpSync(staticAssets, path.join(target, ".next", "static"), { recursive: true });

// electron-builder omits nested directories literally named `node_modules`
// from extra resources. Keep the standalone server dependencies in a separate
// resource folder and expose it through NODE_PATH when the app starts.
fs.cpSync(path.join(standalone, "node_modules"), dependenciesTarget, {
  recursive: true,
});

console.log("Prepared standalone frontend runtime for the desktop package.");
