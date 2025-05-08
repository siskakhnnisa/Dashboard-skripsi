import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np

# --- Config Streamlit ---
st.set_page_config(page_title="Sign Language Hijaiyyah Detection", layout="centered")
st.markdown("<h1 style='text-align: center;'>🖐️ Sign Language Hijaiyyah Detection</h1>", unsafe_allow_html=True)
st.write("Unggah gambar atau gunakan kamera untuk mendeteksi huruf hijaiyyah dari bahasa isyarat Arab menggunakan model CNN berbasis EfficientNet.")

# --- Define Transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load Model ---
model_path = "model3.pth"

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 31)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model(model_path)

# --- Label Mapping ---
labels_list = [
    'Ha (ه)', 'Ain (ع)', 'Alif (ا)', 'Alif Lam (ال)', 'Ba (ب)', 'Dal (د)', 'Dhad (ض)', 'Dzal (ذ)', 'Fa (ف)',
    'Ghain (غ)', 'Ha\' (ح)', 'Jim (ج)', 'Kaf (ك)', 'Kha (خ)', 'Lam (ل)', 'Lam Alif (لا)', 'Mim (م)', 'Nun (ن)',
    'Qaf (ق)', 'Ra (ر)', 'Shad (ص)', 'Sin (س)', 'Syin (ش)', 'Ta (ت)', 'Ta Marbuta (ة)', 'Tha (ط)', 'Tsa (ث)',
    'Wau (و)', 'Ya (ي)', 'Zay (ز)', 'Zha (ظ)'
]

# --- Prediction Function ---
def predict(image):
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item(), probs.squeeze()

# --- Input Gambar: Upload atau Kamera ---
st.markdown("### 📥 Pilih metode input gambar:")

uploaded_file = st.file_uploader("📤 Upload gambar huruf isyarat", type=['jpg', 'jpeg', 'png'])
if uploaded_file is None:
    uploaded_file = st.camera_input("📸 Atau ambil gambar langsung dari kamera")

# --- Proses Prediksi ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Centered preview menggunakan kolom
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='📸 Gambar yang digunakan', width=300)

    if st.button("🔍 Lakukan Prediksi"):
        with st.spinner('⏳ Sedang memproses...'):
            label_idx, confidence, probabilities = predict(image)
            label_predicted = labels_list[label_idx]

            # Tampilkan hasil prediksi utama
            st.markdown("---")
            st.markdown(
                f"<h3 style='text-align: center;'>✅ Prediksi: "
                f"<span style='color:#2E86C1'>{label_predicted.upper()}</span></h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='text-align:center;'>📊 <b>Confidence:</b> {confidence*100:.2f}%</p>",
                unsafe_allow_html=True
            )

            # Tampilkan Top-5 Prediksi
            st.markdown("---")
            st.subheader("🏅 Top-5 Prediksi Lain:")
            top5_prob, top5_idx = torch.topk(probabilities, 5)

            for i in range(5):
                label = labels_list[top5_idx[i]]
                prob = top5_prob[i].item() * 100
                st.write(f"**{label.upper()}** - Confidence: {prob:.2f}%")
                st.progress(min(int(prob), 100))

