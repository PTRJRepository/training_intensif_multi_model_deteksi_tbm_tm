import os
import subprocess

def build():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(current_dir, "frontend")
    dist_dir = os.path.join(frontend_dir, "dist")

    print("[BUILD] Building Frontend...")
    try:
        subprocess.run("npm run build", cwd=frontend_dir, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Build failed: {e}")
        return

    print(f"[BUILD] Frontend ready at: {dist_dir}")
    print("[BUILD] Backend will serve the latest files directly from frontend/dist")

if __name__ == "__main__":
    build()
