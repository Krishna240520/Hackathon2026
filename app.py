"""
app.py

Flask backend for the BreedScan frontend.

- Serves frontend/index.html at "/"
- Exposes POST /api/analyze, which accepts an uploaded image, runs the
  Stage 1 species gate + Stage 2 breed classifier from scripts/inference.py,
  and returns the result as JSON.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000 in a browser.

NOTE: This only works once you've trained the three models (see README.md)
and placed the resulting checkpoints in models/species_model.pt,
models/cattle_breed_model.pt and models/buffalo_breed_model.pt. Until then,
/api/analyze returns a clear 503 error explaining what's missing, instead of
crashing or faking a result.
"""

import sys
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from inference import classify_image  # noqa: E402  (import after sys.path tweak)

MODELS_DIR = BASE_DIR / "models"
SPECIES_MODEL = MODELS_DIR / "species_model.pt"
CATTLE_MODEL = MODELS_DIR / "cattle_breed_model.pt"
BUFFALO_MODEL = MODELS_DIR / "buffalo_breed_model.pt"

UPLOAD_DIR = BASE_DIR / "tmp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB, matches the frontend's own check

app = Flask(__name__, static_folder=str(BASE_DIR / "frontend"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    models_ready = SPECIES_MODEL.exists() and CATTLE_MODEL.exists() and BUFFALO_MODEL.exists()
    return jsonify({"status": "ok", "models_ready": models_ready})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Attach it under the 'image' field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Please upload a PNG or JPEG image."}), 400

    if not (SPECIES_MODEL.exists() and CATTLE_MODEL.exists() and BUFFALO_MODEL.exists()):
        return jsonify({
            "error": (
                "Models are not trained yet. Run the three `python scripts/train.py ...` "
                "commands from the README and place the resulting .pt checkpoints in models/."
            )
        }), 503

    threshold = request.form.get("threshold", 0.6)
    try:
        threshold = float(threshold)
    except ValueError:
        threshold = 0.6

    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    file.save(save_path)

    try:
        result = classify_image(
            save_path,
            SPECIES_MODEL,
            CATTLE_MODEL,
            BUFFALO_MODEL,
            threshold=threshold,
        )
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500
    finally:
        save_path.unlink(missing_ok=True)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
