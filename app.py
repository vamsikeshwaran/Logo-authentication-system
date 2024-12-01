import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model('model.h5')


def preprocess_image(img):
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    if prediction[0][0] < 1:
        return "Fake"
    else:
        return "Genuine"


st.title('Image Authenticity Prediction')
st.write("Upload an image to check if it's Fake or Genuine")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    prediction = predict_image(img)
    st.title("Prediction:")
    st.write(prediction)
