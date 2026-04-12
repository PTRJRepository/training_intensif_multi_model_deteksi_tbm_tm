from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_bounds
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "sawit-auto-labeling-secret-key-2026"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = "auto_labeling/uploads"
EXPORTS_FOLDER = "auto_labeling/exports"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif"}

# YOLO Model Configuration - using exp2_yolo11s_1280 model
YOLO_MODEL_PATH = os.path.join(
    "Train", "runs", "exp2_yolo11s_1280", "weights", "best.pt"
)
yolo_model = None


def load_yolo_model():
    """Load YOLO model for auto-labeling"""
    global yolo_model
    if yolo_model is None:
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"YOLO model loaded from {YOLO_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            yolo_model = YOLO("yolo11s.pt")  # Fallback to default model
            logger.info("Loaded default YOLO11s model")
    return yolo_model


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["EXPORTS_FOLDER"] = EXPORTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORTS_FOLDER, exist_ok=True)
os.makedirs("auto_labeling/static/temp", exist_ok=True)

# Global storage for session data (replace with database for production)
session_data = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_image_info(image_path):
    """Get image metadata including geospatial info if available"""
    try:
        with rasterio.open(image_path) as src:
            return {
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs) if src.crs else None,
                "transform": list(src.transform) if src.transform else None,
                "bounds": src.bounds if src.bounds else None,
            }
    except:
        # Fallback for regular images
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "crs": None,
                "transform": None,
                "bounds": None,
            }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify(
                {"error": "Invalid file type. Allowed: PNG, JPG, JPEG, TIFF"}
            ), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(filepath)

        # Get image info
        image_info = get_image_info(filepath)

        # Convert to base64 for preview
        with open(filepath, "rb") as img_file:
            base64_string = base64.b64encode(img_file.read()).decode()

        # Initialize session data
        session_id = f"session_{timestamp}"
        session_data[session_id] = {
            "image_path": filepath,
            "image_info": image_info,
            "original_filename": file.filename,
            "points": [],
            "labels": ["tree", "building", "road", "water"],  # Default categories
            "created_at": datetime.now().isoformat(),
        }

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "image_data": f"data:image/{filename.split('.')[-1].lower()};base64,{base64_string}",
                "image_info": image_info,
                "labels": session_data[session_id]["labels"],
            }
        )

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/add_point", methods=["POST"])
def add_point():
    try:
        data = request.json
        session_id = data["session_id"]
        x = data["x"]
        y = data["y"]
        label = data.get("label", "tree")

        if session_id not in session_data:
            return jsonify({"error": "Session not found"}), 404

        point_data = {
            "id": len(session_data[session_id]["points"]),
            "x": x,
            "y": y,
            "label": label,
            "timestamp": datetime.now().isoformat(),
        }

        session_data[session_id]["points"].append(point_data)

        return jsonify({"success": True, "point": point_data})

    except Exception as e:
        logger.error(f"Add point error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/delete_point", methods=["POST"])
def delete_point():
    try:
        data = request.json
        session_id = data["session_id"]
        point_id = data["point_id"]

        if session_id not in session_data:
            return jsonify({"error": "Session not found"}), 404

        points = session_data[session_id]["points"]
        session_data[session_id]["points"] = [p for p in points if p["id"] != point_id]

        # Update IDs for consistency
        for i, point in enumerate(session_data[session_id]["points"]):
            point["id"] = i

        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"Delete point error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/update_point", methods=["POST"])
def update_point():
    try:
        data = request.json
        session_id = data["session_id"]
        point_id = data["point_id"]
        updates = data["updates"]  # {x, y, label}

        if session_id not in session_data:
            return jsonify({"error": "Session not found"}), 404

        points = session_data[session_id]["points"]
        for point in points:
            if point["id"] == point_id:
                point.update(updates)
                point["timestamp"] = datetime.now().isoformat()
                return jsonify({"success": True, "point": point})

        return jsonify({"error": "Point not found"}), 404

    except Exception as e:
        logger.error(f"Update point error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/get_points/<session_id>")
def get_points(session_id):
    try:
        if session_id not in session_data:
            return jsonify({"error": "Session not found"}), 404

        return jsonify(
            {
                "success": True,
                "points": session_data[session_id]["points"],
                "labels": session_data[session_id]["labels"],
            }
        )

    except Exception as e:
        logger.error(f"Get points error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/add_label", methods=["POST"])
def add_label():
    try:
        data = request.json
        session_id = data["session_id"]
        new_label = data["label"]

        if session_id not in session_data:
            return jsonify({"error": "Session not found"}), 404

        if new_label not in session_data[session_id]["labels"]:
            session_data[session_id]["labels"].append(new_label)

        return jsonify({"success": True, "labels": session_data[session_id]["labels"]})

    except Exception as e:
        logger.error(f"Add label error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/export/<format>/<session_id>")
def export_data(format, session_id):
    """Export labeled data to various formats"""
    try:
        if session_id not in session_data:
            return jsonify({"error": "Session not found"}), 404

        session = session_data[session_id]
        points = session["points"]
        image_info = session["image_info"]

        if not points:
            return jsonify({"error": "No points to export"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "csv":
            return export_csv(points, session["original_filename"], timestamp)
        elif format == "geojson":
            return export_geojson(
                points, image_info, session["original_filename"], timestamp
            )
        elif format == "shapefile":
            return export_shapefile(
                points, image_info, session["original_filename"], timestamp
            )
        else:
            return jsonify({"error": "Unsupported format"}), 400

    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def export_csv(points, original_filename, timestamp):
    """Export to CSV format"""
    df = pd.DataFrame(points)
    filename = f"labeling_{timestamp}.csv"
    filepath = os.path.join(app.config["EXPORTS_FOLDER"], filename)

    df.to_csv(filepath, index=False)
    return send_file(filepath, as_attachment=True, download_name=filename)


def export_geojson(points, image_info, original_filename, timestamp):
    """Export to GeoJSON format"""
    features = []

    for point in points:
        # Convert pixel coordinates to geographic coordinates if available
        if image_info and "transform" in image_info and image_info["transform"]:
            transform = image_info["transform"]
            geo_x = transform[0] + transform[1] * point["x"] + transform[2] * point["y"]
            geo_y = transform[3] + transform[4] * point["x"] + transform[5] * point["y"]
            geometry = Point(geo_x, geo_y)
        else:
            # Use pixel coordinates with warning
            geometry = Point(point["x"], point["y"])

        feature = {
            "type": "Feature",
            "geometry": geometry.__geo_interface__,
            "properties": {
                "id": point["id"],
                "label": point["label"],
                "pixel_x": point["x"],
                "pixel_y": point["y"],
                "timestamp": point["timestamp"],
            },
        }
        features.append(feature)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "image_filename": original_filename,
            "export_timestamp": timestamp,
            "crs": image_info.get("crs", "Unknown")
            if image_info
            else "Pixel Coordinates",
        },
    }

    filename = f"labeling_{timestamp}.geojson"
    filepath = os.path.join(app.config["EXPORTS_FOLDER"], filename)

    with open(filepath, "w") as f:
        json.dump(feature_collection, f, indent=2)

    return send_file(filepath, as_attachment=True, download_name=filename)


def export_shapefile(points, image_info, original_filename, timestamp):
    """Export to Shapefile format"""
    # Convert points to GeoDataFrame
    geometries = []
    properties = []

    for point in points:
        if image_info and "transform" in image_info and image_info["transform"]:
            # Convert to geographic coordinates
            transform = image_info["transform"]
            geo_x = transform[0] + transform[1] * point["x"] + transform[2] * point["y"]
            geo_y = transform[3] + transform[4] * point["x"] + transform[5] * point["y"]
            geometries.append(Point(geo_x, geo_y))
        else:
            geometries.append(Point(point["x"], point["y"]))

        properties.append(
            {
                "id": point["id"],
                "label": point["label"],
                "pixel_x": point["x"],
                "pixel_y": point["y"],
                "timestamp": point["timestamp"],
            }
        )

    gdf = gpd.GeoDataFrame(properties, geometry=geometries)

    # Set CRS if available
    if image_info and "crs" in image_info and image_info["crs"]:
        gdf.crs = image_info["crs"]
    else:
        # Default to WEB Mercator if no CRS info
        gdf.crs = "EPSG:3857"

    filename_base = f"labeling_{timestamp}"
    export_path = os.path.join(app.config["EXPORTS_FOLDER"], filename_base)

    # Export shapefile (creates multiple files)
    gdf.to_file(export_path)

    # Create a ZIP file containing all shapefile components
    import zipfile

    zip_filename = f"{filename_base}.zip"
    zip_path = os.path.join(app.config["EXPORTS_FOLDER"], zip_filename)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(export_path):
            file_path = os.path.join(export_path, file)
            zipf.write(file_path, file)

    return send_file(zip_path, as_attachment=True, download_name=zip_filename)


@app.route("/sessions")
def list_sessions():
    """List all active sessions"""
    sessions_info = {}
    for sid, session in session_data.items():
        sessions_info[sid] = {
            "original_filename": session["original_filename"],
            "points_count": len(session["points"]),
            "created_at": session["created_at"],
        }
    return jsonify(sessions_info)


@app.route("/auto_label", methods=["POST"])
def auto_label():
    """Auto-label image using YOLO model from exp2_yolo11s_1280"""
    try:
        data = request.json
        session_id = data.get("session_id")
        conf_threshold = data.get("conf_threshold", 0.25)
        iou_threshold = data.get("iou_threshold", 0.45)

        if session_id and session_id in session_data:
            image_path = session_data[session_id]["image_path"]
        elif "image_path" in data:
            image_path = data["image_path"]
        else:
            return jsonify({"error": "No image path or session provided"}), 400

        # Load model
        model = load_yolo_model()

        # Run inference
        results = model(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # Parse detections
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes[i].xyxy[0].tolist()
                conf = float(boxes[i].conf[0])
                cls_id = int(boxes[i].cls[0])
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "label": cls_name,
                    "confidence": conf,
                    "bbox": {
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                    },
                    "center": {
                        "x": (box[0] + box[2]) / 2,
                        "y": (box[1] + box[3]) / 2,
                    }
                })

        # Add detections as points in session
        if session_id and session_id in session_data:
            for det in detections:
                point_data = {
                    "id": len(session_data[session_id]["points"]),
                    "x": det["center"]["x"],
                    "y": det["center"]["y"],
                    "label": det["label"],
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "auto_labeled": True,
                    "timestamp": datetime.now().isoformat(),
                }
                session_data[session_id]["points"].append(point_data)

        return jsonify({
            "success": True,
            "detections": detections,
            "total": len(detections),
        })

    except Exception as e:
        logger.error(f"Auto-label error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
