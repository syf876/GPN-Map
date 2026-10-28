"""Run both local services. Stop with Ctrl+C."""
import os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if __name__=="__main__":
    os.chdir(ROOT)
    children=[]
    try:
        children.append(subprocess.Popen([sys.executable,"-m","uvicorn","api_round2:app","--host","127.0.0.1","--port","8000"]))
        children.append(subprocess.Popen([sys.executable,"frontend/server.py"]))
        print("Open http://127.0.0.1:8080 ; Ctrl+C to stop.",flush=True)
        while all(p.poll() is None for p in children):
            try: children[0].wait(timeout=1)
            except subprocess.TimeoutExpired: pass
    except KeyboardInterrupt: pass
    finally:
        for p in children:
            if p.poll() is None: p.terminate()
        for p in children: p.wait()
