"""
inference.py

Hierarchical prediction:

    1. Stage 1 (species gate) predicts cattle / buffalo / other.
    2. If the top prediction is "other", OR the model's confidence is below
       --threshold, the image is rejected as not cattle or buffalo.
    3. Otherwise, the matching Stage 2 breed model predicts the sub-breed.

Usage:
    python scripts/inference.py --image path/to/photo.jpg
    python scripts/inference.py --image path/to/photo.jpg --threshold 0.6 --json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import build_model, get_input_size

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NOT_RECOGNIZED_MESSAGE = "Species not recognized as Cattle or Buffalo"


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["num_classes"], arch=ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    input_size = get_input_size(ckpt["arch"])
    return model, idx_to_class, input_size


def preprocess(image_path: Path, input_size: int, device):
    tf = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


@torch.no_grad()
def predict(model, idx_to_class, tensor):
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    top_prob, top_idx = probs.max(dim=0)
    return idx_to_class[top_idx.item()], top_prob.item(), probs


def classify_image(image_path, species_model_path, cattle_breed_model_path,
                    buffalo_breed_model_path, threshold=0.6, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = Path(image_path)

    species_model, species_classes, species_input_size = load_checkpoint(Path(species_model_path), device)
    species_tensor = preprocess(image_path, species_input_size, device)
    species_label, species_conf, _ = predict(species_model, species_classes, species_tensor)

    result = {
        "image": str(image_path),
        "species": species_label,
        "species_confidence": round(species_conf, 4),
    }

    if species_label == "other" or species_conf < threshold:
        result["status"] = "rejected"
        result["message"] = NOT_RECOGNIZED_MESSAGE
        result["breed"] = None
        result["breed_confidence"] = None
        return result

    breed_model_path = cattle_breed_model_path if species_label == "cattle" else buffalo_breed_model_path
    breed_model, breed_classes, breed_input_size = load_checkpoint(Path(breed_model_path), device)
    breed_tensor = preprocess(image_path, breed_input_size, device)
    breed_label, breed_conf, _ = predict(breed_model, breed_classes, breed_tensor)

    result["status"] = "classified"
    result["message"] = f"{species_label.capitalize()} — {breed_label} breed"
    result["breed"] = breed_label
    result["breed_confidence"] = round(breed_conf, 4)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--species_model", default="models/species_model.pt")
    ap.add_argument("--cattle_breed_model", default="models/cattle_breed_model.pt")
    ap.add_argument("--buffalo_breed_model", default="models/buffalo_breed_model.pt")
    ap.add_argument("--threshold", type=float, default=0.6,
                     help="Minimum species-gate confidence required to accept cattle/buffalo")
    ap.add_argument("--json", action="store_true", help="Print raw JSON instead of a formatted summary")
    args = ap.parse_args()

    result = classify_image(
        args.image, args.species_model, args.cattle_breed_model,
        args.buffalo_breed_model, threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Image: {result['image']}")
    if result["status"] == "rejected":
        print(f"Result: {result['message']}  (species-gate confidence: {result['species_confidence']:.2%})")
    else:
        print(f"Species: {result['species'].capitalize()}  (confidence: {result['species_confidence']:.2%})")
        print(f"Breed:   {result['breed']}  (confidence: {result['breed_confidence']:.2%})")


if __name__ == "__main__":
    main()
