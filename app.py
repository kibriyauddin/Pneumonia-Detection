import streamlit as st
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from torchvision import transforms
# Try to import grad-cam, fallback if not available
try:
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    try:
        from grad_cam import EigenCAM
        from grad_cam.utils.image import show_cam_on_image
        GRADCAM_AVAILABLE = True
    except ImportError:
        GRADCAM_AVAILABLE = False
        st.warning("⚠️ GradCAM library not available. Heatmap visualization will be disabled.")
import torch.nn.functional as F
import requests
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .normal-prediction {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .pneumonia-prediction {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .confidence-box {
        background-color: #e2e3e5;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .download-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive using file ID"""
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle large files that require confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
            break
    
    # Save the file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    
    return True

def download_models():
    """Download model files from Google Drive if they don't exist"""
    
    # Model file configurations
    models_config = {
        "swin_transformer_weights.pth": {
            "file_id": "1Tzlr3zIf1iNzBCLHlkYVel7EKtq_tUZF",  # Extract from your first URL
            "size_mb": "~400MB"
        },
        "swin_transformer_full_model.pth": {
            "file_id": "1RsZxwzmIOO5ErpZwOfyrZXpBafUBgmY2",  # Extract from your second URL
            "size_mb": "~400MB"
        }
    }
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    download_status = {}
    
    for model_name, config in models_config.items():
        model_path = models_dir / model_name
        
        if not model_path.exists():
            st.info(f"📥 Downloading {model_name} ({config['size_mb']})...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Connecting to Google Drive...")
                progress_bar.progress(25)
                
                status_text.text("Downloading model file...")
                progress_bar.progress(50)
                
                download_file_from_google_drive(config["file_id"], model_path)
                
                progress_bar.progress(100)
                status_text.text("✅ Download completed!")
                
                download_status[model_name] = "✅ Downloaded successfully"
                
            except Exception as e:
                download_status[model_name] = f"❌ Download failed: {str(e)}"
                st.error(f"Failed to download {model_name}: {str(e)}")
                
        else:
            download_status[model_name] = "✅ Already exists"
    
    return download_status

@st.cache_resource
def load_model():
    """Load the trained Swin Transformer model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load the weights file first (smaller file)
        weights_path = Path("models/swin_transformer_weights.pth")
        full_model_path = Path("models/swin_transformer_full_model.pth")
        
        model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=2)
        
        if weights_path.exists():
            # Load state dict
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            st.success("✅ Loaded model from weights file")
        elif full_model_path.exists():
            # Load full model
            model = torch.load(full_model_path, map_location=device)
            st.success("✅ Loaded full model file")
        else:
            st.error("❌ No model files found. Please download them first.")
            return None, None
        
        model.to(device)
        model.eval()
        
        return model, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess the input image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def predict_image(model, image, device):
    """Make prediction on the input image"""
    class_names = ['NORMAL', 'PNEUMONIA']
    
    input_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    # Get probabilities for both classes
    class_probabilities = {
        class_names[i]: probabilities[0][i].item() * 100 
        for i in range(len(class_names))
    }
    
    return predicted_class, confidence_score, class_probabilities

def generate_cam_visualization(model, image, device):
    """Generate EigenCAM visualization"""
    if not GRADCAM_AVAILABLE:
        st.warning("GradCAM visualization is not available due to missing dependencies.")
        return None, None
        
    try:
        # Prepare target layers (last transformer block)
        target_layers = [model.layers[-1].blocks[-1].norm1]
        cam = EigenCAM(model=model, target_layers=target_layers)
        
        input_tensor = preprocess_image(image).to(device)
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        
        # Convert PIL to numpy for visualization
        rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        return cam_image, rgb_img
    except Exception as e:
        st.error(f"Error generating CAM visualization: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Chest X-Ray Pneumonia Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">AI-Powered Medical Image Analysis - Production Version</h2>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Project Information")
        st.info("""
        **BTech Final Year Project**
        
        🎯 **Objective**: Automated pneumonia detection in chest X-rays using Swin Transformer
        
        🤖 **Model**: Swin Transformer (Base)
        📊 **Accuracy**: 94%
        🏥 **Classes**: Normal vs Pneumonia
        
        📈 **Performance Metrics**:
        - Precision: 94%
        - Recall: 94%
        - F1-Score: 94%
        """)
        
        st.header("📖 How to Use")
        st.markdown("""
        1. Download models (if needed)
        2. Upload a chest X-ray image
        3. Click 'Analyze Image'
        4. View prediction results
        5. Examine AI attention map
        """)
        
        st.header("⚠️ Medical Disclaimer")
        st.warning("""
        This is an educational project and should NOT be used for actual medical diagnosis. 
        Always consult qualified healthcare professionals.
        """)
    
    # Model Download Section
    st.header("📦 Model Setup")
    
    # Check if models exist
    weights_path = Path("models/swin_transformer_weights.pth")
    full_model_path = Path("models/swin_transformer_full_model.pth")
    
    if not weights_path.exists() and not full_model_path.exists():
        st.warning("⚠️ Model files not found. Please download them first.")
        
        if st.button("📥 Download Models from Google Drive", type="primary"):
            with st.spinner("Downloading models... This may take a few minutes..."):
                download_status = download_models()
                
                # Display download status
                st.subheader("📊 Download Status")
                for model_name, status in download_status.items():
                    if "✅" in status:
                        st.success(f"{model_name}: {status}")
                    else:
                        st.error(f"{model_name}: {status}")
                
                st.rerun()  # Refresh the app after download
    else:
        st.success("✅ Model files are available!")
        
        # Display model info
        col1, col2 = st.columns(2)
        with col1:
            if weights_path.exists():
                size_mb = weights_path.stat().st_size / (1024 * 1024)
                st.info(f"📁 **Weights file**: {size_mb:.1f} MB")
        with col2:
            if full_model_path.exists():
                size_mb = full_model_path.stat().st_size / (1024 * 1024)
                st.info(f"📁 **Full model**: {size_mb:.1f} MB")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please ensure model files are downloaded correctly.")
        st.stop()
    
    st.success(f"✅ Model loaded successfully! Running on: {device}")
    
    # File upload
    st.header("📤 Upload Chest X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption="Original Chest X-Ray", use_column_width=True)
        
        with col2:
            st.subheader("📊 Analysis Results")
            
            # Analysis button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image... Please wait"):
                    # Make prediction
                    predicted_class, confidence_score, class_probabilities = predict_image(model, image, device)
                    
                    # Generate CAM visualization
                    cam_image, original_resized = generate_cam_visualization(model, image, device)
                    
                    # Store results in session state
                    st.session_state.predicted_class = predicted_class
                    st.session_state.confidence_score = confidence_score
                    st.session_state.class_probabilities = class_probabilities
                    st.session_state.cam_image = cam_image
                    st.session_state.original_resized = original_resized
                    st.session_state.analysis_done = True
            
            # Display results if analysis has been done
            if hasattr(st.session_state, 'analysis_done') and st.session_state.analysis_done:
                # Display prediction
                if st.session_state.predicted_class == "NORMAL":
                    st.markdown(
                        f'<div class="prediction-box normal-prediction">✅ Prediction: {st.session_state.predicted_class}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box pneumonia-prediction">⚠️ Prediction: {st.session_state.predicted_class}</div>',
                        unsafe_allow_html=True
                    )
                
                # Display confidence scores
                st.subheader("🎯 Confidence Scores")
                for class_name, prob in st.session_state.class_probabilities.items():
                    st.markdown(
                        f'<div class="confidence-box">{class_name}: {prob:.2f}%</div>',
                        unsafe_allow_html=True
                    )
                    st.progress(prob / 100)
        
        # CAM Visualization (only show if analysis has been done and gradcam is available)
        if (hasattr(st.session_state, 'analysis_done') and st.session_state.analysis_done and 
            GRADCAM_AVAILABLE and st.session_state.cam_image is not None):
            st.header("🔥 AI Attention Heatmap (EigenCAM)")
            st.info("The heatmap shows which areas of the X-ray the AI model focused on when making its prediction.")
            
            col3, col4 = st.columns([1, 1])
            
            with col3:
                st.subheader("Original (Resized)")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(st.session_state.original_resized)
                ax.axis('off')
                ax.set_title("Original Image", fontsize=14, fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            with col4:
                st.subheader("Attention Heatmap")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(st.session_state.cam_image)
                ax.axis('off')
                ax.set_title("EigenCAM Visualization", fontsize=14, fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            # Additional insights
            st.header("🧠 AI Insights")
            if st.session_state.predicted_class == "PNEUMONIA":
                st.error("""
                **⚠️ Pneumonia Detected**
                
                The AI model has identified patterns consistent with pneumonia in this chest X-ray. 
                The highlighted areas in the heatmap show where the model detected concerning features.
                
                **Important**: This is an AI prediction and should be verified by a qualified radiologist.
                """)
            else:
                st.success("""
                **✅ Normal Classification**
                
                The AI model has classified this chest X-ray as normal, meaning no obvious signs 
                of pneumonia were detected in the analyzed regions.
                
                **Important**: This does not rule out other conditions and should not replace professional medical evaluation.
                """)
    
    # Model Information Section
    st.header("🤖 Model Architecture & Performance")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("Model Type", "Swin Transformer")
        st.metric("Input Size", "224×224")
    
    with col6:
        st.metric("Test Accuracy", "94%")
        st.metric("Dataset Size", "5,856 images")
    
    with col7:
        st.metric("Classes", "2")
        st.metric("Training Epochs", "10")
    
    # Technical Details
    with st.expander("📋 Technical Details"):
        st.markdown("""
        ### Model Architecture
        - **Base Model**: Swin Transformer (swin_base_patch4_window7_224)
        - **Pre-training**: ImageNet
        - **Fine-tuning**: Chest X-ray pneumonia dataset
        - **Optimizer**: AdamW
        - **Learning Rate**: 1e-4
        - **Batch Size**: 32
        
        ### Dataset
        - **Source**: Kaggle Chest X-Ray Pneumonia Dataset
        - **Training Images**: 5,216
        - **Test Images**: 624
        - **Validation Images**: 16
        - **Classes**: Normal, Pneumonia
        
        ### Performance Metrics
        - **Overall Accuracy**: 94%
        - **Normal Precision**: 92%
        - **Pneumonia Precision**: 95%
        - **Normal Recall**: 92%
        - **Pneumonia Recall**: 95%
        
        ### Production Features
        - **Automatic Model Download**: Downloads models from Google Drive on first run
        - **Model Caching**: Uses Streamlit caching for faster subsequent loads
        - **Error Handling**: Comprehensive error handling for production stability
        - **Memory Management**: Proper cleanup of matplotlib figures
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h3>🎓 BTech Final Year Project - Production Version</h3>
        <p><strong>Chest X-Ray Pneumonia Detection using Swin Transformer</strong></p>
        <p>This project demonstrates the application of state-of-the-art computer vision techniques in medical image analysis.</p>
        <p><em>⚠️ For educational purposes only. Not intended for clinical use.</em></p>
        <p><small>🔗 Models are automatically downloaded from Google Drive on first run</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()