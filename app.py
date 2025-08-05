import streamlit as st
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from pathlib import Path
import gdown

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);}
    .sub-header {font-size: 1.5rem; color: #ff7f0e; margin-bottom: 1rem;}
    .prediction-box {padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center; font-size: 1.2rem; font-weight: bold;}
    .normal-prediction {background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb;}
    .pneumonia-prediction {background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb;}
    .confidence-box {background-color: #e2e3e5; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;}
    .footer {text-align: center; margin-top: 3rem; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)


# ✅ Function to download from Google Drive
def download_file_from_google_drive_gdown(file_id, destination):
    """Download file from Google Drive using gdown (with confirmation bypass)."""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=False, fuzzy=True)
        return destination.exists() and destination.stat().st_size > 1000000
    except Exception as e:
        st.error(f"Error with gdown: {str(e)}")
        return False


# ✅ Download model if not available
def download_models():
    models_config = {
        "swin_transformer_weights.pth": {
            "file_id": "1Tzlr3zIf1iNzBCLHlkYVel7EKtq_tUZF",
            "size_mb": "~400MB",
            "direct_url": "https://drive.google.com/file/d/1Tzlr3zIf1iNzBCLHlkYVel7EKtq_tUZF/view?usp=sharing"
        }
    }

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    download_status = {}

    for model_name, config in models_config.items():
        model_path = models_dir / model_name

        if not model_path.exists():
            st.warning(f"📥 Downloading {model_name} ({config['size_mb']})...")
            success = download_file_from_google_drive_gdown(config["file_id"], model_path)

            if success:
                st.success(f"✅ {model_name} downloaded successfully!")
                download_status[model_name] = "✅ Downloaded successfully"
            else:
                st.error(f"❌ Failed to download {model_name}. Please check the Google Drive link.")
                download_status[model_name] = "❌ Download failed"
        else:
            download_status[model_name] = "✅ Already exists"

    return download_status


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path("models/swin_transformer_weights.pth")

    if not weights_path.exists():
        st.warning("⚠️ Model weights not found. Downloading now...")
        download_models()

    if not weights_path.exists() or weights_path.stat().st_size < 1000000:
        st.error("❌ Model file is missing or corrupted. Please retry download.")
        return None, None

    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=2)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0)


def predict_image(model, image, device):
    class_names = ['NORMAL', 'PNEUMONIA']
    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return class_names[predicted.item()], confidence.item() * 100, {
        class_names[i]: probabilities[0][i].item() * 100 for i in range(len(class_names))
    }


# ✅ Main App
def main():
    st.markdown('<h1 class="main-header">🫁 Chest X-Ray Pneumonia Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">AI-Powered Medical Image Analysis</h2>', unsafe_allow_html=True)

    # Sidebar Information
    with st.sidebar:
        st.header("📋 Project Information")
        st.info("""
        **BTech Final Year Project**

        🎯 Objective: Pneumonia detection in chest X-rays using Swin Transformer
        🤖 Model: Swin Transformer (Base)
        📊 Accuracy: 94%
        🏥 Classes: Normal vs Pneumonia
        """)

        st.warning("""
        ⚠️ This is an educational project. 
        Not for actual medical diagnosis.
        """)

    # Download Model
    st.header("📦 Model Setup")
    download_status = download_models()

    if not any("✅" in s for s in download_status.values()):
        st.stop()

    st.success("✅ Model is ready!")

    # Load Model
    model, device = load_model()
    if model is None:
        st.stop()

    st.success(f"✅ Model loaded successfully! Running on: {device}")

    # File upload
    st.header("📤 Upload Chest X-Ray Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing..."):
                predicted_class, confidence, class_probs = predict_image(model, image, device)

                if predicted_class == "NORMAL":
                    st.markdown(f'<div class="prediction-box normal-prediction">✅ Prediction: {predicted_class}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box pneumonia-prediction">⚠️ Prediction: {predicted_class}</div>', unsafe_allow_html=True)

                st.subheader("🎯 Confidence Scores")
                for cls, prob in class_probs.items():
                    st.markdown(f'<div class="confidence-box">{cls}: {prob:.2f}%</div>', unsafe_allow_html=True)
                    st.progress(prob / 100)


if __name__ == "__main__":
    main()
