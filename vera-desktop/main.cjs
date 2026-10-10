const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

let backend;
let frontend;

function waitForService(url, attempts = 120) {
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const request = http.get(url, (response) => {
        response.resume();
        if (response.statusCode >= 200 && response.statusCode < 500) {
          resolve();
        } else {
          retry();
        }
      });
      request.on("error", retry);

      function retry() {
        if (attempts-- <= 0) {
          reject(new Error(`VERA did not start: ${url}`));
          return;
        }
        setTimeout(attempt, 500);
      }
    };
    attempt();
  });
}

function stopServices() {
  backend?.kill();
  frontend?.kill();
  backend = undefined;
  frontend = undefined;
}

async function startVera() {
  const userDataPath = app.getPath("userData");
  fs.mkdirSync(userDataPath, { recursive: true });

  const resourcesPath = app.isPackaged ? process.resourcesPath : path.join(__dirname, "..");
  const backendPath = app.isPackaged
    ? path.join(resourcesPath, "backend", "vera-api")
    : path.join(resourcesPath, "vera-engine", "dist", "vera-api", "vera-api");
  const frontendPath = app.isPackaged
    ? path.join(resourcesPath, "frontend", "server.js")
    : path.join(resourcesPath, "vera-desktop", "frontend-runtime", "server.js");
  const frontendDependenciesPath = app.isPackaged
    ? path.join(resourcesPath, "frontend-deps")
    : path.join(resourcesPath, "vera-desktop", "frontend-deps");

  if (!fs.existsSync(backendPath)) throw new Error(`Backend executable not found:\n${backendPath}`);
  if (!fs.existsSync(frontendPath)) throw new Error(`Frontend server not found:\n${frontendPath}`);

  backend = spawn(backendPath, [], {
    cwd: userDataPath,
    env: {
      ...process.env,
      TALENTLENS_DB_PATH: path.join(userDataPath, "VERA.db"),
      TALENTLENS_FRONTEND_ORIGINS: "http://127.0.0.1:3000,http://localhost:3000",
    },
  });

  frontend = spawn(process.execPath, [frontendPath], {
    cwd: path.dirname(frontendPath),
    env: {
      ...process.env,
      ELECTRON_RUN_AS_NODE: "1",
      HOSTNAME: "127.0.0.1",
      NODE_PATH: frontendDependenciesPath,
      PORT: "3000",
    },
  });

  await waitForService("http://127.0.0.1:8000/openapi.json");
  await waitForService("http://127.0.0.1:3000");

  const window = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });
  await window.loadURL("http://127.0.0.1:3000");
}

app.whenReady().then(async () => {
  try {
    await startVera();
  } catch (error) {
    stopServices();
    dialog.showErrorBox("VERA could not start", error.message || String(error));
    app.quit();
  }
});

app.on("window-all-closed", () => {
  stopServices();
  app.quit();
});
