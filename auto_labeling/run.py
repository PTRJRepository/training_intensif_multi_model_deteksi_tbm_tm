import subprocess
import sys
import os
import socket

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def main():
    print("🚀 Starting Rebinmas Tree Detection...")
    local_ip = get_ip()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(current_dir, "python_app")
    frontend_dist_dir = os.path.join(current_dir, "frontend", "dist")
    
    try:
        import fastapi
        import uvicorn
        import rasterio
        import ultralytics
        import geopandas
    except ImportError as e:
        print(f"❌ Missing dependency: {e.name}")
        print("Please run: pip install fastapi uvicorn rasterio ultralytics geopandas shapely pillow")
        return

    print(f"📍 Server root: {app_dir}")
    print(f"🎨 Frontend dist: {frontend_dist_dir}")
    print(f"🌍 App is available at:")
    print(f"   - Local:   http://localhost:8000")
    print(f"   - Network: http://{local_ip}:8000")
    print("💡 Press Ctrl+C to stop.")
    
    sys.path.insert(0, app_dir)
    
    import uvicorn
    try:
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", 
            port=8000,
            reload=True,
            reload_dirs=[app_dir, frontend_dist_dir],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Tool stopped.")

if __name__ == "__main__":
    main()
