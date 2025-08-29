# Peach Leaf Disease Classifier 🍑🦠

End-to-end example project for classifying **healthy vs. diseased** peach leaves using PyTorch,
with a **Streamlit** web UI.

- Train a ResNet-18 on your dataset (folder-per-class).
- Export a lightweight `.pt` model.
- Serve interactive predictions with Streamlit.
- CI smoke test on push.
- Ready for GitHub: sensible `.gitignore`, MIT license, docs, and scripts.

> Recreated on 2025-08-29 from scratch.

## Repo Structure

```
peach-disease-classifier/
├─ app/streamlit_app.py
├─ models/train.py
├─ models/infer.py
├─ models/dataset.py
├─ scripts/download_data.py
├─ tests/test_smoke.py
├─ data/               # put your images here (not tracked)
│  ├─ train/Healthy/...
│  ├─ train/Diseased/...
│  └─ val/{Healthy,Diseased}/...
├─ weights/            # saved model weights (gitignored)
├─ .github/workflows/ci.yml
├─ .gitignore
├─ requirements.txt
├─ setup.cfg
├─ Dockerfile
├─ LICENSE
└─ README.md
```

## Quickstart

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Put data
Expect **ImageFolder** layout (class-per-folder). Example:
```
data/train/Healthy/...jpg
data/train/Diseased/...jpg
data/val/Healthy/...jpg
data/val/Diseased/...jpg
```

### 3) Train
```bash
python -m models.train --data_dir data --epochs 10 --lr 1e-3 --batch_size 16 --img_size 224 --out weights/model.pt
```

### 4) Inference
```bash
python -m models.infer --weights weights/model.pt --image path/to/leaf.jpg
```

### 5) Streamlit app
```bash
streamlit run app/streamlit_app.py -- --weights weights/model.pt
```

### 6) Docker (optional)
```bash
docker build -t peach-disease .
docker run -p 8501:8501 -v $(pwd)/weights:/app/weights peach-disease streamlit run app/streamlit_app.py -- --weights weights/model.pt
```

## GitHub: Push this project

```bash
# in the project root
git init
git add .
git commit -m "feat: initial commit — peach leaf disease classifier"
git branch -M main
git remote add origin https://github.com/barannyaprakk/peach-disease-classifier.git
git push -u origin main
```

> İpucu: `data/` ve `weights/` klasörleri `.gitignore` içindedir; büyük dosyaları Git'e göndermeyin. Örnek görseller eklemek isterseniz az sayıda küçük imaj koyun.
