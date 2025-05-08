# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
# from PIL import Image
# import numpy as np

# # --- Config Streamlit ---
# st.set_page_config(page_title="Sign Language Hijaiyyah Detection", layout="centered")
# st.title("🖐️ Sign Language Hijaiyyah Detection with EfficientNet")
# st.write("Upload gambar huruf hijaiyyah dari bahasa isyarat Arab untuk dideteksi.")

# # --- Define Transformation  ---
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # --- Load Model ---
# model_path = "model2.pth" 

# @st.cache_resource(show_spinner=False)
# def load_model(model_path):
#     weights = EfficientNet_B0_Weights.DEFAULT
#     model = efficientnet_b0(weights=weights)
    
#     # Modify classifier
#     num_ftrs = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(num_ftrs, 31)
    
#     # Load saved checkpoint
#     checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     return model

# model = load_model(model_path)

# # --- Label Mapping ---
# labels_list = [
#     'Ha', 'ain', 'alif', 'alif_lam', 'ba', 'dal', 'dhad', 'dzal', 'fa',
#     'ghain', 'ha', 'jim', 'kaf', 'kha', 'lam', 'lam_alif', 'mim', 'nun',
#     'qaf', 'ra', 'shad', 'sin', 'syin', 'ta', 'ta_marbuta', 'tha', 'tsa',
#     'wau', 'ya', 'zay', 'zha'
# ]



# # --- Prediction Function ---
# def predict(image):
#     img = transform(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         outputs = model(img)
#         probs = torch.softmax(outputs, dim=1)
#         conf, pred = torch.max(probs, 1)
#     return pred.item(), conf.item(), probs.squeeze()

# # --- Upload Gambar ---
# uploaded_file = st.file_uploader("Upload gambar huruf isyarat", type=['jpg', 'jpeg', 'png'])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Gambar yang diupload', use_column_width=True)
    
#     if st.button("🔍 Prediksi"):
#         with st.spinner('Sedang mendeteksi...'):
#             label_idx, confidence, probabilities = predict(image)
#             label_predicted = labels_list[label_idx]
            
#             st.success(f"**Prediksi: {label_predicted.upper()}**")
#             st.info(f"Confidence: {confidence*100:.2f}%")
            
#             # Optional: tampilkan Top-5 prediction
#             st.subheader("Top-5 Prediction:")
#             top5_prob, top5_idx = torch.topk(probabilities, 5)
#             for i in range(5):
#                 st.write(f"{labels_list[top5_idx[i]]}: {top5_prob[i]*100:.2f}%")
#     st.markdown(f"### Prediksi: {predicted_label}")
#     st.markdown(f"Confidence: {confidence:.2f}%")


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
st.write("Unggah gambar huruf hijaiyyah dari bahasa isyarat Arab untuk dideteksi oleh model CNN berbasis EfficientNet.")

# --- Define Transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load Model ---
model_path = "model2.pth"

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
    'Ha', 'ain', 'alif', 'alif_lam', 'ba', 'dal', 'dhad', 'dzal', 'fa',
    'ghain', 'ha', 'jim', 'kaf', 'kha', 'lam', 'lam_alif', 'mim', 'nun',
    'qaf', 'ra', 'shad', 'sin', 'syin', 'ta', 'ta_marbuta', 'tha', 'tsa',
    'wau', 'ya', 'zay', 'zha'
]

# --- Prediction Function ---
def predict(image):
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item(), probs.squeeze()

# --- Upload Gambar ---
uploaded_file = st.file_uploader("📤 Upload gambar huruf isyarat", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Menampilkan gambar di tengah
    st.image(image, caption='📸 Gambar yang diupload', use_container_width=True)

    if st.button("🔍 Lakukan Prediksi"):
        with st.spinner('⏳ Sedang memproses...'):
            label_idx, confidence, probabilities = predict(image)
            label_predicted = labels_list[label_idx]

            # Tampilan hasil prediksi
            st.markdown("---")
            st.markdown(f"<h3 style='text-align: center;'>✅ Prediksi: <span style='color:#2E86C1'>{label_predicted.upper()}</span></h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>📊 <b>Confidence:</b> {confidence*100:.2f}%</p>", unsafe_allow_html=True)

            # Top-5 prediction visual
            st.markdown("---")
            st.subheader("🏅 Top-5 Prediksi Lain:")
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            
            for i in range(5):
                label = labels_list[top5_idx[i]]
                prob = top5_prob[i].item() * 100
                st.write(f"**{label.upper()}** - Confidence: {prob:.2f}%")
                st.progress(min(int(prob), 100))  # Batasi ke 100%

