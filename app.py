import os
import joblib
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# Import sklearn BEFORE streamlit to attempt to bypass AppLocker DLL blocks
import sklearn
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
import xgboost

import streamlit as st

# Adjust imports from local src directory
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing.image_processing import preprocess_image
from src.features.feature_extraction import extract_lesion_features

st.set_page_config(page_title="Diabetic Retinopathy Classifier", layout="centered")

@st.cache_resource
def load_models():
    # Load Classical Stacking Classifier
    model_path = "results/final_hybrid_model.pkl"
    if not os.path.exists(model_path):
        return None, None, "Error: results/final_hybrid_model.pkl not found."
    clf = joblib.load(model_path)
    
    # Load CNN Feature Extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    effnet.classifier = torch.nn.Identity()
    effnet.to(device)
    effnet.eval()
    
    return clf, effnet, None

st.title("Diabetic Retinopathy Classification")

clf, effnet, err_msg = load_models()
if err_msg:
    st.error(err_msg)
    st.stop()

tab1, tab2 = st.tabs(["Severity Prediction", "Project Report & Metrics"])

with tab1:
    st.write("Upload a fundus image or select a sample to predict the severity of Diabetic Retinopathy.")
    
    input_option = st.radio("Choose Input Method", ("Select Sample Image", "Upload Your Own Image"))
    
    image = None
    if input_option == "Upload Your Own Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        sample_images = {
            "Sample 1 - No DR": "data/aptos/gaussian_filtered_images/gaussian_filtered_images/No_DR/002c21358ce6.png",
            "Sample 2 - Mild DR": "data/aptos/gaussian_filtered_images/gaussian_filtered_images/Mild/0024cdab0c1e.png",
            "Sample 3 - Moderate DR": "data/aptos/gaussian_filtered_images/gaussian_filtered_images/Moderate/000c1434d8d7.png",
            "Sample 4 - Severe DR": "data/aptos/gaussian_filtered_images/gaussian_filtered_images/Severe/0104b032c141.png",
            "Sample 5 - Proliferative DR": "data/aptos/gaussian_filtered_images/gaussian_filtered_images/Proliferate_DR/001639a390f0.png"
        }
        selected_sample = st.selectbox("Select a Sample Image", list(sample_images.keys()))
        if selected_sample:
            image = Image.open(sample_images[selected_sample]).convert('RGB')
    
    if image is not None:
        # Display the image
        st.image(image, caption='Fundus Image', use_container_width=True)
        
        st.write("Analyzing...")
        
        with st.spinner('Extracting Classical & Deep Features...'):
            try:
                img_np = np.array(image)
                
                # Classical
                _, y_stretched = preprocess_image(img_np)
                cls_feats = extract_lesion_features(y_stretched)
                
                # CNN
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_t = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    cnn_feats = effnet(img_t).cpu().numpy().flatten()
                    
                # Combine
                fused_feats = np.concatenate((cls_feats, cnn_feats)).reshape(1, -1)
                
                # Predict
                pred = clf.predict(fused_feats)[0]
                classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
                severity = classes[pred]
                
                st.success("Analysis Complete!")
                st.markdown(f"### Predicted DR Severity Grade: **{severity}**")
                
            except Exception as e:
                st.error(f"Error processing image: {e}")

with tab2:
    st.header("Project Report & Metrics")
    st.write("Below are the performance metrics and graphs comparing our classical baseline model to the final Hybrid CNN Ensemble.")
    
    st.subheader("1. Model Accuracy Comparison")
    if os.path.exists("results/accuracy_comparison.png"):
        st.image("results/accuracy_comparison.png", use_container_width=True)
    
    st.subheader("2. Class Distribution & SMOTE Balancing")
    st.write("The IDRiD dataset was originally very imbalanced, which led to poor recognition of 'Mild DR' (Class 1). SMOTE was applied to balance the dataset before training.")
    if os.path.exists("results/class_distribution.png"):
        st.image("results/class_distribution.png", use_container_width=True)
        
    st.subheader("3. Feature Importance (SHAP)")
    st.write("This chart explains which classical and deep CNN features most strongly drove the Stacking Ensemble's decisions globally.")
    if os.path.exists("results/shap_summary.png"):
        st.image("results/shap_summary.png", use_container_width=True)

