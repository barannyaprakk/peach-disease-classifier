
import streamlit as st
import torch
from PIL import Image
from models.infer import load_model, predict_image

st.set_page_config(page_title="Peach Leaf Disease Classifier", page_icon="🍑", layout="centered")

st.title("🍑 Peach Leaf Disease Classifier")
st.write("Healthy vs. Diseased peach leaves — upload an image and get a prediction.")

weights = st.sidebar.text_input("Weights path", value="weights/model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def _load(weights_path):
    return load_model(weights_path, device)

model = None
classes = None
try:
    model, tfm, classes, img_size = _load(weights)
    st.sidebar.success(f"Model loaded — classes: {classes}")
except Exception as e:
    st.sidebar.error(f"Load failed: {e}")

upl = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if upl and model:
    img = Image.open(upl).convert("RGB")
    st.image(img, caption="Uploaded", use_container_width=True)
    with st.spinner("Predicting..."):
        idx, probs = predict_image(model, tfm, upl, device)
    st.subheader(f"Prediction: **{classes[idx]}**")
    st.write({c: float(f"{p:.4f}") for c, p in zip(classes, probs)})
    st.caption("This model is a demo; verify results before making decisions.")
else:
    st.info("Upload an image to get started.")
