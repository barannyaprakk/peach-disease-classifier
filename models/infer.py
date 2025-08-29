
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models
from torch import nn

def load_model(weights_path: str, device):
    ckpt = torch.load(weights_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, tfm, classes, img_size

@torch.no_grad()
def predict_image(model, tfm, image_path, device):
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1).cpu().numpy().ravel()
    pred_idx = int(probs.argmax())
    return pred_idx, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tfm, classes, _ = load_model(args.weights, device)
    idx, probs = predict_image(model, tfm, args.image, device)
    print(f"Pred: {classes[idx]}  Probs: {dict(zip(classes, [float(f'{p:.4f}') for p in probs]))}")

if __name__ == "__main__":
    main()
