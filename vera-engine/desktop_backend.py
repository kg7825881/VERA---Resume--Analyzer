import os
from pathlib import Path
import uvicorn

data_root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VERA"
data_root.mkdir(parents=True, exist_ok=True)

os.chdir(data_root)
os.environ["TALENTLENS_DB_PATH"] = str(data_root / "VERA.db")
os.environ.setdefault("TALENTLENS_FRONTEND_ORIGINS", "http://127.0.0.1:3000")

from api import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")