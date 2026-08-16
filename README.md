# Cattle vs. Buffalo Breed Classifier

A two-stage deep learning pipeline that:

1. **Stage 1 (Species Gate):** Classifies an input image as `cattle`, `buffalo`, or `other`
   (i.e. *not cattle or buffalo*).
2. **Stage 2 (Breed Classifier):** If the image is `cattle` or `buffalo`, a dedicated
   breed classifier identifies the specific sub-breed (e.g. Gir, Sahiwal, Murrah, Jaffarabadi).

If Stage 1 is not confident the animal is cattle or buffalo, the pipeline returns a
professional rejection message instead of forcing a guess — it does **not** fall back to
Stage 2 in that case.

---

## 1. Getting a Dataset

You said you need help sourcing data. A few good starting points (verify each license
before commercial use):

| Source | Contents | Link |
|---|---|---|
| Indian Bovine Breeds (Kaggle) | Broad Indian cattle + buffalo breed image set | kaggle.com/datasets/lukex9442/indian-bovine-breeds |
| Indian Cattle Image Dataset (Kaggle) | 50 ICAR-recognized Indian cattle breeds w/ metadata | kaggle.com/datasets/atharvadarpude/indian-cattle-image-dataset |
| Cows and Buffalo CV Dataset (Kaggle) | Mixed cow/buffalo images | kaggle.com/datasets/raghavdharwal/cows-and-buffalo-computer-vision-dataset |
| Cattle Breeds Dataset (Kaggle) | Smaller, 5 cattle breeds | kaggle.com/datasets/anandkumarsahu09/cattle-breeds-dataset |
| Cow Breed Classification Dataset (Kaggle) | Cattle breed images | kaggle.com/datasets/zaidworks0508/cow-breed-classification-dataset |
| SIH25004 (GitHub) | Reference pipeline built on the Indian Bovine Breeds dataset — useful for sanity-checking folder layout | github.com/ramkamal452/SIH25004 |

**For the `other` (negative) class**, you need images of animals/scenes that are *not*
cattle or buffalo, so the model learns to say "no" instead of always picking the closest
match. Good sources: general animal datasets (e.g. Open Images / ImageNet subsets) for
horses, goats, sheep, dogs, deer, camels, plus some empty-background/farm-scene images.
Aim for at least as many `other` images as your average per-breed count.

## 2. Expected Folder Layout

Place raw images like this before running `data_prep.py`:

```
data/raw/
├── cattle/
│   ├── Gir/            *.jpg
│   ├── Sahiwal/
│   ├── Holstein_Friesian/
│   ├── Jersey/
│   └── ...
├── buffalo/
│   ├── Murrah/
│   ├── Jaffarabadi/
│   ├── Nili_Ravi/
│   ├── Mehsana/
│   └── ...
└── other/
    └── *.jpg            (any non-cattle/buffalo images, flat folder)
```

Folder names become the class labels, so name them exactly as you want them reported.

## 3. Pipeline

```bash
pip install -r requirements.txt

# 1) Split raw data into train/val/test for both stages
python scripts/data_prep.py --raw_dir data/raw --out_dir data/splits

# 2) Train the Stage 1 species gate (cattle / buffalo / other)
python scripts/train.py --data_dir data/splits/species --output models/species_model.pt --epochs 15

# 3) Train the Stage 2 cattle breed classifier
python scripts/train.py --data_dir data/splits/breed_cattle --output models/cattle_breed_model.pt --epochs 20

# 4) Train the Stage 2 buffalo breed classifier
python scripts/train.py --data_dir data/splits/breed_buffalo --output models/buffalo_breed_model.pt --epochs 20

# 5) Run inference on a new image (CLI)
python scripts/inference.py --image path/to/photo.jpg

# 6) Or launch the web app (frontend + backend together)
python app.py
# then open http://localhost:5000
```

## 4. Project Layout

```
cattle_buffalo_classifier/
├── app.py                  # Flask backend — serves frontend/ and POST /api/analyze
├── frontend/
│   └── index.html          # BreedScan UI (upload, drag & drop, results)
├── scripts/
│   ├── data_prep.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
├── models/                 # trained .pt checkpoints go here (empty until you train)
├── data/raw/{cattle,buffalo,other}/...
├── requirements.txt
└── README.md
```

## 5. Frontend ↔ Backend Wiring

`app.py` serves `frontend/index.html` at `/` and exposes `POST /api/analyze`,
which accepts a multipart `image` field, runs the Stage 1 → Stage 2 pipeline
from `scripts/inference.py`, and returns JSON like:

```json
{
  "status": "classified",
  "species": "cattle",
  "species_confidence": 0.94,
  "breed": "Gir",
  "breed_confidence": 0.81,
  "message": "Cattle — Gir breed"
}
```

or, when the image isn't cattle/buffalo:

```json
{
  "status": "rejected",
  "species": "other",
  "species_confidence": 0.88,
  "breed": null,
  "breed_confidence": null,
  "message": "Species not recognized as Cattle or Buffalo"
}
```

`frontend/index.html`'s "Analyze" button now calls this endpoint with `fetch`
instead of the placeholder random-result simulation it shipped with.

**Important:** `/api/analyze` returns a `503` with a clear explanation until
all three `models/*.pt` checkpoints exist — it will not fabricate a result.
Train the models first (Section 3), then start `app.py`.

## 6. Design Notes

- **Backbone:** EfficientNet-B0 (ImageNet-pretrained), fine-tuned — a good accuracy/speed
  trade-off for this image count; swap in `--arch resnet50` for a heavier option.
- **Why a separate `other` class instead of only a confidence threshold?** A softmax
  classifier trained only on cattle/buffalo will *always* pick one of those two, even for
  a goat or a dog, often with high confidence. Training on explicit negative examples
  teaches the model what "neither" looks like. The confidence threshold at inference is a
  second safety net on top of that, catching cases the model is simply unsure about.
- **Two breed models instead of one combined model:** keeps each classifier's decision
  boundary simpler (cattle breeds vs. cattle breeds only) and lets you extend one species'
  breed list without retraining the other.
- **Output for non-cattle/buffalo images:** the inference script returns
  `"Species not recognized as Cattle or Buffalo"` rather than a forced breed guess.
