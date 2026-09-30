const { app, BrowserWindow, dialog, shell } = require("electron");

const { spawn } = require("child_process");

const path = require("path");

const fs = require("fs");

const http = require("http");
 
const OLLAMA_URL = "http://127.0.0.1:11434";
 
// Required for VERA's Browser Semantic flow:

// - qwen2.5:3b-instruct: resume extraction

// - nomic-embed-text: semantic embeddings

const REQUIRED_OLLAMA_MODELS = [

  "qwen2.5:3b-instruct",

  "nomic-embed-text",

];
 
let backend = null;

let frontend = null;

let ollamaProcess = null;

let backendFailure = null;

let frontendFailure = null;
 
function waitForApp(url, attempts = 120) {

  return new Promise((resolve, reject) => {

    const attempt = () => {

      if (backendFailure) {

        reject(new Error(`Backend failed to start:\n${backendFailure}`));

        return;

      }
 
      if (frontendFailure) {

        reject(new Error(`Frontend failed to start:\n${frontendFailure}`));

        return;

      }
 
      const request = http.get(url, (response) => {

        response.resume();
 
        if (response.statusCode >= 200 && response.statusCode < 500) {

          resolve();

          return;

        }
 
        retry();

      });
 
      request.on("error", retry);
 
      function retry() {

        if (attempts-- <= 0) {

          reject(new Error(`VERA did not start at:\n${url}`));

          return;

        }
 
        setTimeout(attempt, 500);

      }

    };
 
    attempt();

  });

}
 
function ollamaJson(pathname) {

  return new Promise((resolve, reject) => {

    const request = http.get(`${OLLAMA_URL}${pathname}`, (response) => {

      let body = "";
 
      response.setEncoding("utf8");

      response.on("data", (chunk) => {

        body += chunk;

      });
 
      response.on("end", () => {

        if (response.statusCode < 200 || response.statusCode >= 300) {

          reject(new Error(`Ollama returned HTTP ${response.statusCode}.`));

          return;

        }
 
        try {

          resolve(JSON.parse(body));

        } catch {

          reject(new Error("Ollama returned an invalid response."));

        }

      });

    });
 
    request.on("error", reject);

  });

}
 
async function ollamaIsRunning() {

  try {

    await ollamaJson("/api/tags");

    return true;

  } catch {

    return false;

  }

}
 
function commandExists(command) {

  return new Promise((resolve) => {

    const process = spawn("where.exe", [command], {

      windowsHide: true,

      stdio: "ignore",

    });
 
    process.on("error", () => resolve(false));

    process.on("close", (code) => resolve(code === 0));

  });

}
 
async function findOllamaExecutable() {

  const localAppData = process.env.LOCALAPPDATA || "";
 
  const standardInstallPath = path.join(

    localAppData,

    "Programs",

    "Ollama",

    "ollama.exe"

  );
 
  if (fs.existsSync(standardInstallPath)) {

    return standardInstallPath;

  }
 
  if (await commandExists("ollama")) {

    return "ollama";

  }
 
  return null;

}
 
async function waitForOllama(attempts = 60) {

  for (let attempt = 0; attempt < attempts; attempt += 1) {

    if (await ollamaIsRunning()) {

      return;

    }
 
    await new Promise((resolve) => setTimeout(resolve, 500));

  }
 
  throw new Error("Ollama did not start in time.");

}
 
async function startOllamaIfNeeded(ollamaExecutable, logStream) {

  if (await ollamaIsRunning()) {

    logStream.write("[Ollama] Already running. VERA will not stop it on exit.\n");

    return;

  }
 
  ollamaProcess = spawn(ollamaExecutable, ["serve"], {

    windowsHide: true,

    stdio: ["ignore", "pipe", "pipe"],

  });
 
  ollamaProcess.stdout.on("data", (data) => {

    logStream.write(`[Ollama] ${data}`);

  });
 
  ollamaProcess.stderr.on("data", (data) => {

    logStream.write(`[Ollama ERROR] ${data}`);

  });
 
  await waitForOllama();

  logStream.write("[Ollama] Started by VERA.\n");

}
 
function pullOllamaModel(ollamaExecutable, model, logStream) {

  return new Promise((resolve, reject) => {

    const process = spawn(ollamaExecutable, ["pull", model], {

      windowsHide: true,

      stdio: ["ignore", "pipe", "pipe"],

    });
 
    let errorText = "";
 
    process.stdout.on("data", (data) => {

      logStream.write(`[Ollama pull ${model}] ${data}`);

    });
 
    process.stderr.on("data", (data) => {

      errorText += data.toString();

      logStream.write(`[Ollama pull ${model} ERROR] ${data}`);

    });
 
    process.on("error", reject);
 
    process.on("close", (code) => {

      if (code === 0) {

        resolve();

      } else {

        reject(

          new Error(

            `Could not download Ollama model "${model}".\n${errorText}`

          )

        );

      }

    });

  });

}
 
function modelIsInstalled(installedModels, requiredModel) {

  return installedModels.some((model) => {

    const name = model.name || model.model || "";

    return name === requiredModel || name.startsWith(`${requiredModel}:`);

  });

}
 
async function ensureOllamaForBrowserSemantic(logStream) {

  const ollamaExecutable = await findOllamaExecutable();
 
  if (!ollamaExecutable) {

    const answer = await dialog.showMessageBox({

      type: "warning",

      title: "Ollama is required",

      message: "VERA requires Ollama for local resume processing and Browser Semantic analysis.",

      detail:

        "Would you like to download the official Ollama installer now?\n\nAfter installation, reopen VERA.",

      buttons: ["Download Ollama", "Cancel"],

      defaultId: 0,

      cancelId: 1,

      noLink: true,

    });
 
    if (answer.response === 0) {

      await shell.openExternal("https://ollama.com/download/OllamaSetup.exe");

    }
 
    throw new Error(

      "Install Ollama, then reopen VERA to complete the local AI setup."

    );

  }
 
  await startOllamaIfNeeded(ollamaExecutable, logStream);
 
  const tags = await ollamaJson("/api/tags");

  const installedModels = tags.models || [];
 
  const missingModels = REQUIRED_OLLAMA_MODELS.filter(

    (model) => !modelIsInstalled(installedModels, model)

  );
 
  if (missingModels.length === 0) {

    logStream.write("[Ollama] All required VERA models are already installed.\n");

    return;

  }
 
  const answer = await dialog.showMessageBox({

    type: "question",

    title: "Download VERA local AI models?",

    message: "VERA needs local Ollama models before it can start.",

    detail:

      `The following models will be downloaded:\n\n${missingModels.join(

        "\n"

      )}\n\nThis may take several minutes and uses disk space.`,

    buttons: ["Download models", "Cancel"],

    defaultId: 0,

    cancelId: 1,

    noLink: true,

  });
 
  if (answer.response !== 0) {

    throw new Error(

      "VERA needs the required Ollama models before it can start."

    );

  }
 
  for (const model of missingModels) {

    await pullOllamaModel(ollamaExecutable, model, logStream);

  }
 
  logStream.write("[Ollama] Required VERA models downloaded successfully.\n");

}
 
function stopServices() {

  if (backend && !backend.killed) {

    backend.kill();

  }
 
  if (frontend && !frontend.killed) {

    frontend.kill();

  }
 
  backend = null;

  frontend = null;

}
 
function stopOllamaStartedByVera() {

  if (!ollamaProcess || !ollamaProcess.pid) {

    return;

  }
 
  spawn("taskkill", ["/pid", String(ollamaProcess.pid), "/T", "/F"], {

    windowsHide: true,

    stdio: "ignore",

  });
 
  ollamaProcess = null;

}
 
function attachProcessLogging(process, name, logStream, onFailure) {

  process.stdout.on("data", (data) => {

    logStream.write(`[${name}] ${data}`);

  });
 
  process.stderr.on("data", (data) => {

    logStream.write(`[${name} ERROR] ${data}`);

  });
 
  process.on("error", (error) => {

    const message = `${name} could not launch: ${error.message}`;

    logStream.write(`${message}\n`);

    onFailure(message);

  });
 
  process.on("exit", (code, signal) => {

    if (code !== 0 && code !== null) {

      const message = `${name} stopped unexpectedly (exit code ${code}${

        signal ? `, signal ${signal}` : ""

      }).`;
 
      logStream.write(`${message}\n`);

      onFailure(message);

    }

  });

}
 
async function startVera() {

  backendFailure = null;

  frontendFailure = null;
 
  const userDataPath = app.getPath("userData");

  fs.mkdirSync(userDataPath, { recursive: true });
 
  const logPath = path.join(userDataPath, "vera-startup.log");

  const logStream = fs.createWriteStream(logPath, { flags: "a" });
 
  logStream.write(`\n--- VERA started: ${new Date().toISOString()} ---\n`);
 
  // Ollama is checked before VERA starts its backend/frontend.

  await ensureOllamaForBrowserSemantic(logStream);
 
  const resourcesPath = app.isPackaged

    ? process.resourcesPath

    : path.join(__dirname, "..");
 
  const backendPath = app.isPackaged

    ? path.join(resourcesPath, "backend", "vera-api.exe")

    : path.join(

        resourcesPath,

        "vera-engine",

        "dist",

        "vera-api",

        "vera-api.exe"

      );
 
  const frontendPath = app.isPackaged

    ? path.join(resourcesPath, "frontend", "server.js")

    : path.join(

        resourcesPath,

        "vera-frontend",

        ".next",

        "standalone",

        "server.js"

      );
 
  if (!fs.existsSync(backendPath)) {

    throw new Error(`Backend executable not found:\n${backendPath}`);

  }
 
  if (!fs.existsSync(frontendPath)) {

    throw new Error(`Frontend server not found:\n${frontendPath}`);

  }
 
  backend = spawn(backendPath, [], {

    cwd: userDataPath,

    windowsHide: true,

    env: {

      ...process.env,

      TALENTLENS_DB_PATH: path.join(userDataPath, "VERA.db"),

      TALENTLENS_FRONTEND_ORIGINS:

        "http://127.0.0.1:3000,http://localhost:3000",

    },

  });
 
  attachProcessLogging(backend, "Backend", logStream, (message) => {

    backendFailure = message;

  });
 
  await waitForApp("http://127.0.0.1:8000/openapi.json");
 
  frontend = spawn(process.execPath, [frontendPath], {

    cwd: path.dirname(frontendPath),

    windowsHide: true,

    env: {

      ...process.env,

      ELECTRON_RUN_AS_NODE: "1",

      HOSTNAME: "127.0.0.1",

      PORT: "3000",

      NEXT_PUBLIC_API_BASE_URL: "http://127.0.0.1:8000",

    },

  });
 
  attachProcessLogging(frontend, "Frontend", logStream, (message) => {

    frontendFailure = message;

  });
 
  await waitForApp("http://127.0.0.1:3000");
 
  const mainWindow = new BrowserWindow({

    width: 1440,

    height: 940,

    minWidth: 1000,

    minHeight: 700,

    webPreferences: {

      contextIsolation: true,

      nodeIntegration: false,

    },

  });
 
  await mainWindow.loadURL("http://127.0.0.1:3000");

}
 
app.whenReady().then(async () => {

  try {

    await startVera();

  } catch (error) {

    stopServices();

    stopOllamaStartedByVera();
 
    dialog.showErrorBox(

      "VERA could not start",

      `${error.message || String(error)}\n\nCheck:\n%APPDATA%\\VERA\\vera-startup.log`

    );
 
    app.quit();

  }

});
 
app.on("window-all-closed", () => {

  stopServices();

  stopOllamaStartedByVera();

  app.quit();

});
 