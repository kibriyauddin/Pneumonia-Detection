import streamlit as st
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from torchvision import transforms
import torch.nn.functional as F
import requests
import os
from pathlib import Path
import gdown

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
</style>
""", unsafe_allow_html=True)

def download_file_from_google_drive_gdown(file_id, destination):
    """Download file from Google Drive using gdown library"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=False)
        return True
    except Exception as e:
        st.error(f"Error with gdown: {str(e)}")
        return False

def download_file_from_google_drive_requests(file_id, destination):
    """Download file from Google Drive using requests (fallback method)"""
    try:
        # Try direct download URL first
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check if we got a confirmation page (for large files)
        if "virus scan" in response.text.lower() or "download_warning" in response.text:
            # Find the confirmation token
            for line in response.text.split('\n'):
                if 'confirm=' in line and 'download' in line:
                    import re
                    token = re.search(r'confirm=([a-zA-Z0-9\-_]+)', line)
                    if token:
                        confirm_url = f"https://drive.google.com/uc?export=download&confirm={token.group(1)}&id={file_id}"
                        response = session.get(confirm_url, stream=True)
                        break
        
        # Download the file
        total_size = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        return total_size > 1000000  # Return True if file is larger than 1MB
        
    except Exception as e:
        st.error(f"Error with requests method: {str(e)}")
        return False

def download_models():
    """Download model files from Google Drive if they don't exist"""
    
    # Model file configurations
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
            st.info(f"📥 Downloading {model_name} ({config['size_mb']})...")
            st.info(f"🔗 **Direct Download Link**: {config['direct_url']}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Attempting download method 1 (gdown)...")
                progress_bar.progress(25)
                
                # Try gdown first (more reliable for large files)
                success = download_file_from_google_drive_gdown(config["file_id"], model_path)
                
                if not success or not model_path.exists() or model_path.stat().st_size < 1000000:
                    status_text.text("Attempting download method 2 (requests)...")
                    progress_bar.progress(50)
                    
                    # Try requests method as fallback
                    success = download_file_from_google_drive_requests(config["file_id"], model_path)
                
                if model_path.exists() and model_path.stat().st_size > 1000000:
                    file_size = model_path.stat().st_size
                    progress_bar.progress(100)
                    status_text.text(f"✅ Download completed! ({file_size / (1024*1024):.1f} MB)")
                    download_status[model_name] = "✅ Downloaded successfully"
                else:
                    progress_bar.progress(0)
                    status_text.text("❌ Download failed - please try manual download")
                    download_status[model_name] = "❌ Download failed - try manual download"
                    
                    # Show manual download instructions
                    st.error(f"""
                    **Manual Download Required**
                    
                    The automatic download failed. Please:
                    1. Click this link: {config['direct_url']}
                    2. Download the file manually
                    3. Upload it using the file uploader below
                    """)
                
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
        
        # Try to load the weights file
        weights_path = Path("models/swin_transformer_weights.pth")
        
        if not weights_path.exists():
            st.error("❌ Model weights file not found. Please download it first.")
            return None, None
        
        # Check file size
        file_size = weights_path.stat().st_size
        if file_size < 1000000:  # Less than 1MB
            st.error(f"❌ Model file seems corrupted (only {file_size} bytes). Please re-download.")
            return None, None
        
        # Create model architecture
        model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=2)
        
        # Load state dict
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            st.success("✅ Loaded model from weights file")
        except Exception as e:
            st.error(f"❌ Error loading model weights: {str(e)}")
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

def generate_simple_attention_map(model, image, device):
    """Generate a simple attention visualization using model features"""
    try:
        input_tensor = preprocess_image(image).to(device)
        
        # Hook to capture feature maps
        feature_maps = []
        def hook_fn(module, input, output):
            feature_maps.append(output)
        
        # Register hook on the last layer
        hook = model.layers[-1].register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hook
        hook.remove()
        
        if feature_maps:
            # Get the last feature map and average across channels
            feature_map = feature_maps[-1].squeeze(0)  # Remove batch dimension
            
            # Average across the feature dimension to get spatial attention
            if len(feature_map.shape) == 3:  # [H, W, features] or [features, H, W]
                if feature_map.shape[0] > feature_map.shape[1]:  # [features, H, W]
                    attention_map = torch.mean(feature_map, dim=0)
                else:  # [H, W, features]
                    attention_map = torch.mean(feature_map, dim=-1)
            else:
                attention_map = feature_map
            
            # Normalize and resize
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
            attention_map = F.interpolate(
                attention_map.unsqueeze(0).unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            
            return attention_map.cpu().numpy()
        
        return None
        
    except Exception as e:
        st.error(f"Error generating attention map: {str(e)}")
        return None

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
        1. Download model (if needed)
        2. Upload a chest X-ray image
        3. Click 'Analyze Image'
        4. View prediction results
        5. Examine simple attention map
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
    
    if not weights_path.exists():
        st.warning("⚠️ Model file not found. Please download it first.")
        
        # Show download options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Model from Google Drive", type="primary"):
                with st.spinner("Downloading model... This may take a few minutes..."):
                    download_status = download_models()
                    
                    # Display download status
                    st.subheader("📊 Download Status")
                    for model_name, status in download_status.items():
                        if "✅" in status:
                            st.success(f"{model_name}: {status}")
                        else:
                            st.error(f"{model_name}: {status}")
                    
                    if any("✅" in status for status in download_status.values()):
                        st.success("🔄 Reloading app with downloaded model...")
                        st.rerun()
        
        with col2:
            st.subheader("📤 Manual Upload")
            uploaded_model = st.file_uploader(
                "Upload model file manually", 
                type=['pth'],
                help="If automatic download fails, upload the .pth file here"
            )
            
            if uploaded_model is not None:
                # Save uploaded model
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                
                model_path = models_dir / "swin_transformer_weights.pth"
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.read())
                
                st.success(f"✅ Model uploaded successfully! ({model_path.stat().st_size / (1024*1024):.1f} MB)")
                st.rerun()
    
    else:
        st.success("✅ Model file is available!")
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        st.info(f"📁 **Model file**: {size_mb:.1f} MB")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please ensure model file is downloaded correctly.")
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
                    
                    # Generate simple attention map
                    attention_map = generate_simple_attention_map(model, image, device)
                    
                    # Store results in session state
                    st.session_state.predicted_class = predicted_class
                    st.session_state.confidence_score = confidence_score
                    st.session_state.class_probabilities = class_probabilities
                    st.session_state.attention_map = attention_map
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
        
        # Attention Visualization
        if (hasattr(st.session_state, 'analysis_done') and st.session_state.analysis_done and 
            st.session_state.attention_map is not None):
            
            st.header("🔥 AI Attention Map")
            st.info("This shows areas the AI model focused on when making its prediction.")
            
            col3, col4 = st.columns([1, 1])
            
            with col3:
                st.subheader("Original Image")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(np.array(image.resize((224, 224))))
                ax.axis('off')
                ax.set_title("Original Image", fontsize=14, fontweight='bold')
                st.pyplot(fig)
                plt.close()
            
            with col4:
                st.subheader("Attention Map")
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Overlay attention map on original image
                original_resized = np.array(image.resize((224, 224)))
                im = ax.imshow(original_resized, alpha=0.6)
                heatmap = ax.imshow(st.session_state.attention_map, alpha=0.4, cmap='jet')
                
                ax.axis('off')
                ax.set_title("Attention Heatmap", fontsize=14, fontweight='bold')
                plt.colorbar(heatmap, ax=ax, shrink=0.8)
                st.pyplot(fig)
                plt.close()
            
            # Additional insights
            st.header("🧠 AI Insights")
            if st.session_state.predicted_class == "PNEUMONIA":
                st.error("""
                **⚠️ Pneumonia Detected**
                
                The AI model has identified patterns consistent with pneumonia in this chest X-ray. 
                The attention map shows areas the model focused on when making this prediction.
                
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
        - **Multiple Download Methods**: Automatic download with fallback options
        - **Manual Upload**: Upload model file directly if download fails
        - **Model Validation**: Checks file size and integrity
        - **Error Handling**: Comprehensive error handling for production stability
        - **Simple Attention Map**: Custom attention visualization without external dependencies
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h3>🎓 BTech Final Year Project - Production Version</h3>
        <p><strong>Chest X-Ray Pneumonia Detection using Swin Transformer</strong></p>
        <p>This project demonstrates the application of state-of-the-art computer vision techniques in medical image analysis.</p>
        <p><em>⚠️ For educational purposes only. Not intended for clinical use.</em></p>
        <p><small>🔗 Model can be downloaded automatically or uploaded manually</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()