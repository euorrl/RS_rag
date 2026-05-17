import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402


def main() -> None:
    """启动 RAG Web API 服务。"""
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )


if __name__ == "__main__":
    main()
