import os
import subprocess
import shutil
import sys

def build():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(current_dir, "frontend")
    dist_dir = os.path.join(frontend_dir, "dist")
    static_dir = os.path.join(current_dir, "python_app", "static")

    print("[BUILD] Building Frontend...")
    # Run npm install and build
    try:
        # Use shell=True for Windows compatibility
        subprocess.run("npm install && npm run build", cwd=frontend_dir, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Build failed: {e}")
        return

    print("[BUILD] Copying build to Python app...")
    if os.path.exists(static_dir):
        shutil.rmtree(static_dir)

    shutil.copytree(dist_dir, static_dir)
    print("[BUILD] Frontend updated successfully!")

if __name__ == "__main__":
    build()
