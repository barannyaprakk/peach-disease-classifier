# Peach Leaf Disease Classifier 🍑🦠

End-to-end example project for classifying **healthy vs. diseased** peach leaves using **PyTorch**, with a **Streamlit** web UI.

- Train a ResNet-18 on your dataset (folder-per-class).
- Export a lightweight `.pt` model.
- Serve interactive predictions with Streamlit.
- CI smoke test on push.
- Ready for GitHub: sensible `.gitignore`, MIT license, docs, and scripts.

---

## Academic Context 🎓
This project was originally developed as part of the **Biomedical Signal Processing** course in my 4th year (second semester) at **Osmaniye Korkut Ata University**.  
It demonstrates the use of deep learning (transfer learning with ResNet-18) and computer vision to solve a practical classification problem in agriculture: distinguishing between healthy and diseased peach leaves.

---

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

---

## Quickstart
1) **Environment**
```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) **Prepare data**
This project expects the standard **ImageFolder** layout (one folder per class):
```
data/train/Healthy/...jpg
data/train/Diseased/...jpg
data/val/Healthy/...jpg
data/val/Diseased/...jpg
```
- You can organize **your own dataset** in this structure, or
- Use a **public peach leaf dataset** from the internet (e.g., university repositories, Kaggle-style datasets, or agricultural research datasets).  
  After downloading, simply place the images into the folders above.  
  > Tip: Keep a small **validation** split (`val/`) for unbiased evaluation.

3) **Train**
```bash
python -m models.train --data_dir data --epochs 10 --lr 1e-3 --batch_size 16 --img_size 224 --out weights/model.pt
```

4) **Inference**
```bash
python -m models.infer --weights weights/model.pt --image path/to/leaf.jpg
```

5) **Streamlit app**
```bash
streamlit run app/streamlit_app.py -- --weights weights/model.pt
```

6) **Docker (optional)**
```bash
docker build -t peach-disease .
docker run -p 8501:8501 -v $(pwd)/weights:/app/weights peach-disease \
  streamlit run app/streamlit_app.py -- --weights weights/model.pt
```

---

## Notes on Data & Privacy
- **No raw images are committed** to the repo by design: the `data/` directory is in `.gitignore`.  
- If you want to share sample images, keep them **small and few** (and verify you have the right to redistribute).  

---

## Tech Stack
- Python, PyTorch, torchvision
- Streamlit
- scikit-learn (metrics), Pillow
- GitHub Actions (CI)
- Docker (optional)

---

## License
MIT — see `LICENSE` for details.

---

**Note:** The `data/` and `weights/` directories are listed in `.gitignore`; do not push large files to Git. If you want to include sample images, keep them small and few.
