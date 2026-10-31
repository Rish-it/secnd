import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../../", import.meta.url));
const build = spawnSync("npm", ["run", "build"], { cwd: root, stdio: "inherit" });
if (build.status !== 0) process.exit(build.status ?? 1);
const demo = spawnSync("uv", ["run", "python", "-m", "backend.demo_seed", ...process.argv.slice(2)], {
  cwd: root, stdio: "inherit", env: { ...process.env, UV_NO_SYNC: "true" },
});
process.exit(demo.status ?? 1);
