import streamlit as st
import requests
from PIL import Image
import io
import os

st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")

st.title("CIFAR-10 Image Classifier")

Flask_url = "http://127.0.0.1:5000"
pred_endpoint = Flask_url.rstrip("/") + "/predict"

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            file = {"file": (file.name, file.getvalue(), file.type)}
            try:
                with st.spinner("wait while our model is predicting..."):
                    resp = requests.post(pred_endpoint, files=file)
                if resp.status_code == 200:
                    st.error(f"API Error:{resp.status_code}{resp.text}")
                else:
                    data = resp.json()
                    st.success(f"Predicted Class: **{data['predicted_class']}")
                    st.subheader("Class Probabilities:")
                    probs = data.get("probabilities", {})
                    for cls, prob in probs.items():
                        st.write(f"{cls}: {prob:.3f}")
            except Exception as e:
                st.error(f"Faild to call API: {str(e)}")