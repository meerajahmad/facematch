import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN

# Load models and data
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Bollywood Celebrity Look-alike", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #121212; color: white; }
        .title { text-align: center; font-size: 36px; font-weight: bold; margin-bottom: 10px; }
        .subtitle { text-align: center; font-size: 18px; margin-bottom: 30px; }
        .predict-btn { background-color: #FF4B4B; color: white; border-radius: 10px; font-size: 18px; padding: 10px 20px; }
        .image-container { display: flex; justify-content: center; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">Which Bollywood celebrity do you look like?</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and our deep learning model will find your celebrity twin!</p>', unsafe_allow_html=True)

# Upload image
uploaded_image = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])

# Save uploaded image
def save_uploaded_image(uploaded_image):
    try:
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

# Extract features
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    
    if not results:
        return None  # No face detected
    
    x, y, width, height = results[0]['box']
    face = img_rgb[y:y + height, x:x + width]
    
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    
    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    
    return model.predict(preprocessed_img).flatten()

# Recommend function
def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), feature.reshape(1, -1))[0][0] for feature in feature_list]
    index_pos = sorted(enumerate(similarity), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos, similarity[index_pos]

# Prediction
if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image).resize((300, 300))  # Resize uploaded image
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)

        if features is None:
            st.error("No face detected! Please upload a clear image.")
        else:
            index_pos, confidence = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

            # Load and resize predicted image
            predicted_image = Image.open(filenames[index_pos]).resize((300, 300))

            # Layout with columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Your Uploaded Image")
                st.image(display_image, width=300)

            with col2:
                st.subheader(f"You Look Like {predicted_actor}!")
                st.image(predicted_image, width=300)

                # Progress bar for match confidence
                st.progress(int(confidence * 100))
                st.write(f"Match Confidence: **{int(confidence * 100)}%**")
