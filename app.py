from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import sqlite3
from datetime import datetime
import uuid
import json
from pathlib import Path
from io import BytesIO
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

# =============================
# Config
# =============================
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
DETECTION_FOLDER = os.environ.get("DETECTION_FOLDER", "runs")
DB_NAME = os.environ.get("DB_NAME", "database.db")
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
APP_DEBUG = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
LOCAL_CONFIG_FOLDER = os.environ.get("LOCAL_CONFIG_FOLDER", ".local_config")

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp", "jfif"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Default location (Muzaffarabad, Azad Kashmir)
CITY_NAME = "Muzaffarabad, Azad Kashmir"
CITY_COORDS = (34.3700, 73.4711)

# =============================

os.makedirs(LOCAL_CONFIG_FOLDER, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath(os.path.join(LOCAL_CONFIG_FOLDER, "ultralytics")))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(LOCAL_CONFIG_FOLDER, "matplotlib")))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Try to import YOLO
try:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("Warning: YOLO not loaded:", e)
    model = None

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DETECTION_FOLDER"] = DETECTION_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["DETECTION_FOLDER"], exist_ok=True)


# ===== DB setup =====
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                filename TEXT,
                annotated_filename TEXT,
                media_type TEXT,
                detected_classes TEXT,
                timestamp TEXT,
                location TEXT,
                incharge TEXT
            )"""
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(detections)").fetchall()}
        if "annotated_filename" not in columns:
            conn.execute("ALTER TABLE detections ADD COLUMN annotated_filename TEXT")
        if "media_type" not in columns:
            conn.execute("ALTER TABLE detections ADD COLUMN media_type TEXT")


init_db()


# ===== Upload and detection helpers =====
def allowed_file(filename, allowed_extensions):
    ext = os.path.splitext(filename or "")[1].lower().lstrip(".")
    return ext in allowed_extensions


def save_uploaded_image(file_storage, raw):
    try:
        img = Image.open(BytesIO(raw))
        img.verify()
    except (UnidentifiedImageError, Exception):
        raise ValueError("Uploaded file is not a valid image")

    img = Image.open(BytesIO(raw))
    fmt = img.format or "JPEG"
    fmt_lower = fmt.lower()
    ext_map = {"jpeg": "jpg", "jfif": "jpg"}
    ext = ext_map.get(fmt_lower, fmt_lower)

    orig_name = secure_filename(file_storage.filename or "image")
    base = os.path.splitext(orig_name)[0]
    filename = f"{uuid.uuid4()}_{base}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if ext in ("jpg", "jpeg"):
        img = img.convert("RGB")
        img.save(filepath, format="JPEG", quality=90)
    else:
        img.save(filepath, format=img.format)

    return filepath


def save_uploaded_video(file_storage):
    if not allowed_file(file_storage.filename, ALLOWED_VIDEO_EXTENSIONS):
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video type. Allowed video formats: {allowed}.")

    orig_name = secure_filename(file_storage.filename or "video")
    base, ext = os.path.splitext(orig_name)
    filename = f"{uuid.uuid4()}_{base}{ext.lower()}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(filepath)
    return filepath


def save_uploaded_media(file_storage):
    filename = file_storage.filename or ""
    if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
        raw = file_storage.read()
        file_storage.stream.seek(0)
        return save_uploaded_image(file_storage, raw), "image"
    if allowed_file(filename, ALLOWED_VIDEO_EXTENSIONS):
        return save_uploaded_video(file_storage), "video"

    allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS))
    raise ValueError(f"Unsupported file type. Allowed formats: {allowed}.")


def find_annotated_output(run_dir, original_filename):
    run_path = Path(run_dir)
    if not run_path.exists():
        return None

    original_stem = Path(original_filename).stem
    candidates = [p for p in run_path.rglob("*") if p.is_file()]
    matching = [p for p in candidates if p.stem == original_stem]
    chosen = matching[0] if matching else (candidates[0] if candidates else None)
    if chosen is None:
        return None

    return str(chosen.relative_to(app.config["DETECTION_FOLDER"])).replace("\\", "/")


def run_detection(filepath):
    if model is None:
        return {"model_missing"}, None

    detected_set = set()
    annotated_filename = None
    try:
        run_name = f"detection_{uuid.uuid4().hex}"
        results = model.predict(
            source=filepath,
            save=True,
            project=app.config["DETECTION_FOLDER"],
            name=run_name,
            exist_ok=True,
        )
        names = getattr(model, "names", {}) or {}
        for res in results:
            if getattr(res, "boxes", None) is not None:
                for cls in res.boxes.cls:
                    try:
                        detected_set.add(names[int(cls)])
                    except Exception:
                        detected_set.add(str(cls))
        annotated_filename = find_annotated_output(
            os.path.join(app.config["DETECTION_FOLDER"], run_name),
            os.path.basename(filepath),
        )
    except Exception as e:
        print("Model inference error:", e)

    return detected_set, annotated_filename


def count_detected_classes(rows):
    counts = {"small": 0, "medium": 0, "large": 0}
    for row in rows:
        detected_field = (row[3] or "").lower()
        for size in counts:
            if size in detected_field:
                counts[size] += 1
    return counts


def marker_color(detected_field):
    detected = (detected_field or "").lower()
    if "large" in detected:
        return "#dc3545"
    if "medium" in detected:
        return "#ffc107"
    if "small" in detected:
        return "#28a745"
    return "#6c757d"


# ===== Parse location =====
def parse_location_to_coords(location_text):
    if not location_text:
        return CITY_COORDS
    parts = [p.strip() for p in location_text.split(",")]
    if len(parts) >= 2:
        try:
            lat = float(parts[0])
            lon = float(parts[1])
            return (lat, lon)
        except Exception:
            pass
    if "muzaffarabad" in location_text.lower() or "azad" in location_text.lower():
        return CITY_COORDS
    return CITY_COORDS


# ===== Routes =====
@app.route("/")
def index():
    return render_template(
        "index.html",
        max_upload_mb=MAX_UPLOAD_MB,
        image_extensions=sorted(ALLOWED_IMAGE_EXTENSIONS),
        video_extensions=sorted(ALLOWED_VIDEO_EXTENSIONS),
        allowed_extensions=sorted(ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS),
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/detections/<path:filename>")
def detection_file(filename):
    return send_from_directory(app.config["DETECTION_FOLDER"], filename)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    location = request.form.get("location", "").strip()
    incharge = request.form.get("incharge", "").strip()

    if not file or file.filename == "":
        return "No file uploaded.", 400

    try:
        filepath, media_type = save_uploaded_media(file)
    except ValueError as e:
        return str(e), 400

    detected_set, annotated_filename = run_detection(filepath)
    detected_str = ", ".join(sorted(detected_set)) if detected_set else "None"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """INSERT INTO detections
               (id, filename, annotated_filename, media_type, detected_classes, timestamp, location, incharge)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                os.path.basename(filepath),
                annotated_filename,
                media_type,
                detected_str,
                timestamp,
                location or None,
                incharge or None,
            ),
        )

    return redirect(url_for("analytics"))


@app.route("/analytics")
def analytics():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, filename, annotated_filename, detected_classes, timestamp, location, incharge, media_type
               FROM detections
               ORDER BY timestamp DESC"""
        )
        rows = cursor.fetchall()

    counts = count_detected_classes(rows)
    map_data = []

    for r in rows:
        coords = parse_location_to_coords(r[5] or "")
        map_data.append({
            "id": r[0],
            "filename": r[1],
            "annotated": r[2],
            "detected": r[3],
            "timestamp": r[4],
            "location": r[5],
            "incharge": r[6],
            "media_type": r[7] or "image",
            "lat": coords[0],
            "lon": coords[1],
            "color": marker_color(r[3]),
        })

    return render_template(
        "analytics.html",
        rows=rows,
        chart_data=[counts["small"], counts["medium"], counts["large"]],
        small_count=counts["small"],
        medium_count=counts["medium"],
        large_count=counts["large"],
        map_data_json=json.dumps(map_data),
        city_coords=CITY_COORDS,
        city_name=CITY_NAME,
    )


@app.errorhandler(413)
def file_too_large(_error):
    return f"Uploaded file is too large. Maximum size is {MAX_UPLOAD_MB} MB.", 413


if __name__ == "__main__":
    app.run(debug=APP_DEBUG)
