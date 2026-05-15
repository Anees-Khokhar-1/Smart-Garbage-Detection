# Smart Garbage Detection & Classification using YOLO

A Flask-based web application that detects and classifies garbage as small, medium, or large using a custom YOLO model. The system stores detection results in SQLite and visualizes records with analytics, annotated outputs, and geolocation mapping.

## Project Overview

Users can upload images or supported videos. The system:

- Runs detection using `best.pt` or the path provided by `MODEL_PATH`
- Classifies trash into small, medium, and large categories
- Saves original uploads in `uploads/`
- Saves annotated YOLO outputs in `runs/`
- Stores each detection in a local SQLite database
- Displays analytics with counts, records, result links, and a map

This makes the system suitable for smart city waste monitoring, cleanliness analysis, and field inspection workflows.

## Project Structure

```text
project/
|-- app.py                 # Main Flask application
|-- best.pt                # YOLO trained model
|-- database.db            # SQLite DB, auto-created/migrated
|-- templates/
|   |-- index.html
|   `-- analytics.html
|-- uploads/               # User-uploaded media
|-- runs/                  # Annotated YOLO outputs
|-- static/                # Optional CSS/assets
|-- requirements.txt
`-- README.md
```

## Features

- Garbage detection for small, medium, and large waste
- Secure image validation with Pillow
- Supported video uploads: `mp4`, `avi`, `mov`, `mkv`, `webm`
- Configurable upload limit through `MAX_UPLOAD_MB`
- Annotated detection result links in analytics
- Multi-class counting, so one upload can increment multiple categories
- SQLite logging with filename, annotated filename, media type, classes, timestamp, location, and in-charge
- Leaflet map centered on Muzaffarabad by default

## Configuration

Environment variables:

```bash
MODEL_PATH=best.pt
UPLOAD_FOLDER=uploads
DETECTION_FOLDER=runs
DB_NAME=database.db
MAX_UPLOAD_MB=100
FLASK_DEBUG=0
```

Set `FLASK_DEBUG=1` only during local development.

## Installation & Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes

- Internet access is required for the current CDN-hosted frontend libraries.
- If `best.pt` is missing or cannot load, uploads still log with `model_missing`.
- Enter location as `lat,lon` for precise marker placement. Otherwise, the app falls back to Muzaffarabad, Azad Kashmir.
