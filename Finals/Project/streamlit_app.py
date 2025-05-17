import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import tempfile
from decord import VideoReader, cpu
import numpy as np

# Model Definition
class CustomFineTuneModel(nn.Module):
    def __init__(self, base_model):
        super(CustomFineTuneModel, self).__init__()
        self.base_model = base_model
        self.lstm_layer = nn.LSTM(input_size=1280, hidden_size=64, batch_first=True)
        self.final_classifier = nn.Linear(64, 1)

    def forward(self, x):
        batch_size, sequence, channels, height, width = x.size()
        x = x.view(batch_size * sequence, channels, height, width)
        x = self.base_model(x)
        x = x.view(batch_size, sequence, -1)
        output, (h_n, c_n) = self.lstm_layer(x)
        x = self.final_classifier(h_n.squeeze(0))
        return x.squeeze()

# Load Model
@st.cache_resource
def load_model(weights_path):
    base = models.efficientnet_v2_s(weights='DEFAULT')
    base.classifier = nn.Identity()
    model = CustomFineTuneModel(base)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

# Video Preprocessing
def extract_frames(video_path, num_frames=16, size=(224, 224)):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = [Image.fromarray(vr[i].asnumpy()).resize(size) for i in indices]
    return frames

def preprocess_frames(frames, transform):
    tensors = [transform(img) for img in frames]
    video_tensor = torch.stack(tensors, dim=0)
    video_tensor = video_tensor.unsqueeze(0)
    return video_tensor

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🥱 Fatigue Detection with CNN-LSTM 😴")

uploaded_file = st.file_uploader("Upload a video 📽️ file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    # Layout with two columns: Video and First Frame
    col1, col2 = st.columns([1,5])

    with col1:
        st.video(video_path)
        st.caption("📽️ Uploaded video preview")

    st.write("Extracting 🖼️ frames...")
    frames = extract_frames(video_path, num_frames=24, size=(224, 224))
    st.success(f"✅ Extracted {len(frames)} frames.")

    with col2:
        st.image(frames[0], caption="🖼️ First Frame")

    # Preprocess frames
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    video_tensor = preprocess_frames(frames, transform)

    # Load and run model
    model = load_model("Finals/Project/model2_new_weights.pth")

    with torch.no_grad():
        output = model(video_tensor)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0

        st.metric("🧠 Prediction", "Fatigued" if pred == 1 else "Awake")