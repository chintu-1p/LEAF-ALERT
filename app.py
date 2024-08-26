import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model(r'C:\Users\Lenovo\Downloads\trained_plant_disease_model.keras')

# Class labels for PlantVillage dataset
class_labels = {
    0: 'Apple Scab',
    1: 'Apple Black Rot',
    2: 'Apple Cedar Apple Rust',
    3: 'Apple Healthy',
    4: 'Blueberry Healthy',
    5: 'Cherry Powdery Mildew',
    6: 'Cherry Healthy',
    7: 'Corn Cercospora Leaf Spot Gray Leaf Spot',
    8: 'Corn Common Rust',
    9: 'Corn Northern Leaf Blight',
    10: 'Corn Healthy',
    11: 'Grape Black Rot',
    12: 'Grape Esca (Black Measles)',
    13: 'Grape Leaf Blight (Isariopsis Leaf Spot)',
    14: 'Grape Healthy',
    15: 'Orange Haunglongbing (Citrus Greening)',
    16: 'Peach Bacterial Spot',
    17: 'Peach Healthy',
    18: 'Pepper, Bell Bacterial Spot',
    19: 'Pepper, Bell Healthy',
    20: 'Potato Early Blight',
    21: 'Potato Late Blight',
    22: 'Potato Healthy',
    23: 'Raspberry Healthy',
    24: 'Soybean Healthy',
    25: 'Squash Powdery Mildew',
    26: 'Strawberry Leaf Scorch',
    27: 'Strawberry Healthy',
    28: 'Tomato Bacterial Spot',
    29: 'Tomato Early Blight',
    30: 'Tomato Late Blight',
    31: 'Tomato Leaf Mold',
    32: 'Tomato Septoria Leaf Spot',
    33: 'Tomato Spider Mites Two-Spotted Spider Mite',
    34: 'Tomato Target Spot',
    35: 'Tomato Mosaic Virus',
    36: 'Tomato Yellow Leaf Curl Virus',
    37: 'Tomato Healthy'
}

st.title('Plant Disease Classification')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the image for prediction
    image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    # Predict the class
    if st.button('Predict'):
        prediction = model.predict(input_arr)
        predicted_class = np.argmax(prediction)
        st.write(f'Prediction: {class_labels[predicted_class]}')


