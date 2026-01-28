import streamlit as st
import cv2
import pickle
import numpy as np
from skimage.feature import hog
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Vehicle Type Classification",
    page_icon="🚗",
    layout="wide"
)

IMG_SIZE = (64, 64)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("vehicle_model_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return pipeline, le

pipeline, le = load_model()

# ---------------- FEATURE EXTRACTION ----------------
def extract_hog_features(image):
    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=6,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)

# ---------------- UI HEADER ----------------
st.title("🚗 Vehicle Type Classification System")
st.markdown(
    "Classical Machine Learning deployment using **HOG + RBF SVM**"
)

st.divider()

# ---------------- LAYOUT ----------------
left_col, right_col = st.columns([1.2, 1])

# ---------------- LEFT PANEL (UPLOAD + PREDICT) ----------------
with left_col:
    st.subheader("📤 Upload Vehicle Image")

    uploaded_file = st.file_uploader(
        "Choose an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict Vehicle Type", use_container_width=True):
            with st.spinner("Analyzing image..."):
                features = extract_hog_features(image_np)
                pred_class = pipeline.predict(features)[0]
                confidence = pipeline.predict_proba(features).max()

            st.success("Prediction Complete")

            st.markdown("### ✅ Prediction Result")
            st.metric(
                label="Vehicle Type",
                value=le.inverse_transform([pred_class])[0]
            )

            st.metric(
                label="Confidence",
                value=f"{confidence * 100:.2f}%"
            )

# ---------------- RIGHT PANEL (MODEL INFO) ----------------
with right_col:
    st.subheader("🧠 Model Explanation")

    st.markdown("""
    **Pipeline Overview**
    - Image resized to **64×64**
    - Converted to grayscale
    - **HOG (Histogram of Oriented Gradients)** feature extraction
    - Feature scaling using **StandardScaler**
    - Classification using **RBF SVM**
    
    **Why this model?**
    - Handles high-dimensional image features
    - Captures non-linear shape patterns
    - Works efficiently without deep learning
    
    **Classes Supported**
    - Big Truck
    - City Car
    - Multi Purpose Vehicle
    - Sedan
    - Sport Utility Vehicle
    - Truck
    - Van
    """)

    st.info(
        "This system uses classical machine learning only — "
        "no CNNs or deep learning frameworks."
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption("Developed as a Machine Learning Image Classification Project")
