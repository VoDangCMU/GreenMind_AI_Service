import subprocess
import multiprocessing
from pathlib import Path
import uvicorn

def run_qdrant():
    """Chạy Qdrant bằng Docker Compose"""
    compose_path = Path("qdrant/docker-compose.yml")
    print(f"Starting Qdrant from {compose_path}...")
    subprocess.run(
        ["docker-compose", "-f", str(compose_path), "up", "-d"],
        check=True
    )

def run_fastapi():
    """Chạy FastAPI server"""
    print("Starting FastAPI...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"]
    )

if __name__ == "__main__":
    run_qdrant()

    fastapi_process = multiprocessing.Process(target=run_fastapi)
    fastapi_process.start()
    
    try:
        fastapi_process.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        fastapi_process.terminate()
        subprocess.run(
            ["docker-compose", "-f", "qdrant/docker-compose.yml", "down"],
            check=True
        )