import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import os

# --- Config Streamlit ---
st.set_page_config(page_title="Sign Language Hijaiyyah Detection", layout="centered")
st.title("🖐️ Sign Language Hijaiyyah Detection with EfficientNet")
st.write("Upload gambar huruf hijaiyyah dari bahasa isyarat Arab untuk dideteksi.")

# --- Define Transformation (harus sama dengan training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load Model ---
@st.cache_resource
def load_model(model_path):
    try:
        # Verifikasi file model ada
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {os.path.abspath(model_path)}")
        
        # Load model architecture
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        
        # Modify classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 31)  # Sesuaikan dengan jumlah kelas
        
        # Coba beberapa metode loading untuk kompatibilitas
        try:
            # Coba loading biasa dulu
            state_dict = torch.load(model_path, map_location='cpu')
        except:
            # Coba dengan weights_only=False jika error
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Load state dict dengan strict=False untuk handle mismatch kecil
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()

model_path = "model.pth"
if not os.path.exists(model_path):
    st.error(f"File model tidak ditemukan di: {os.path.abspath(model_path)}")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error inisialisasi model: {str(e)}")
    st.stop()

# --- Label Mapping ---
labels_list = ['ain', 'alif', 'ba', 'dhad', 'dal', 'fa', 'ghain', 'ha', 'Ha', 'jim', 'kaf', 'kha', 'lam', 'mim', 'nun', 'qaf', 
               'ra', 'shad', 'sin', 'syin', 'tha', 'ta', 'ta_marbuta', 'dzal', 'tsa', 'wau', 'ya', 'zha', 'zay', 'alif_lam', 'lam_alif']

# --- Prediction Function ---
def predict(image):
    try:
        img = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        return pred.item(), conf.item(), probs.squeeze()
    except Exception as e:
        st.error(f"Error selama prediksi: {str(e)}")
        return None, None, None

# --- Upload Gambar ---
uploaded_file = st.file_uploader("Upload gambar huruf isyarat", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang diupload', use_column_width=True)
        
        if st.button("🔍 Prediksi"):
            with st.spinner('Sedang mendeteksi...'):
                label_idx, confidence, probabilities = predict(image)
                
                if label_idx is not None:
                    label_predicted = labels_list[label_idx]
                    
                    st.success(f"**Prediksi: {label_predicted.upper()}**")
                    st.info(f"Confidence: {confidence*100:.2f}%")
                    
                    # Tampilkan Top-5 prediction
                    st.subheader("Top-5 Prediction:")
                    top5_prob, top5_idx = torch.topk(probabilities, 5)
                    for i in range(5):
                        st.write(f"{labels_list[top5_idx[i]]}: {top5_prob[i]*100:.2f}%")
                else:
                    st.error("Gagal melakukan prediksi")
                    
    except Exception as e:
        st.error(f"Error memproses gambar: {str(e)}")
