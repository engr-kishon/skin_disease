import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import time

# Load the trained model
model_path = 'final_model.keras'
loaded_model = tf.keras.models.load_model(model_path)

# Define class names and their descriptions
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_descriptions = {
    'akiec': {
        'full_name': 'Actinic Keratoses and Intraepithelial Carcinoma / Bowen\'s disease',
        'causes': 'Primarily caused by prolonged exposure to ultraviolet (UV) radiation from the sun or tanning beds.',
        'next_steps': 'Consult a dermatologist for assessment and treatment options which may include cryotherapy, topical medications, or photodynamic therapy.',
        'info': 'These lesions are considered precancerous and can potentially progress to squamous cell carcinoma if left untreated.'
    },
    'bcc': {
        'full_name': 'Basal Cell Carcinoma',
        'causes': 'Typically caused by long-term exposure to UV radiation from sunlight.',
        'next_steps': 'Seek evaluation by a dermatologist. Treatment options often include surgical removal, topical medications, or radiation therapy.',
        'info': 'Basal cell carcinoma is the most common type of skin cancer but rarely spreads to other parts of the body. Early detection and treatment are important.'
    },
    'bkl': {
        'full_name': 'Benign Keratosis-like Lesions',
        'causes': 'Often related to aging and sun exposure.',
        'next_steps': 'Usually benign and require no treatment unless they become irritated or cosmetically concerning. If in doubt, consult a dermatologist.',
        'info': 'These are non-cancerous skin growths that include seborrheic keratoses and solar lentigines.'
    },
    'df': {
        'full_name': 'Dermatofibroma',
        'causes': 'The exact cause is unknown but may be related to minor skin injuries or insect bites.',
        'next_steps': 'Generally harmless and do not require treatment. If bothersome, they can be surgically removed.',
        'info': 'Dermatofibromas are benign skin nodules that often appear on the legs and arms. They are firm to the touch and may be itchy or tender.'
    },
    'mel': {
        'full_name': 'Melanoma',
        'causes': 'Caused by DNA damage to skin cells, usually from UV radiation, leading to mutations and uncontrolled cell growth.',
        'next_steps': 'Immediate consultation with a dermatologist is critical. Treatment may include surgery, immunotherapy, radiation therapy, or chemotherapy.',
        'info': 'Melanoma is a serious form of skin cancer that can spread to other parts of the body. Early detection and treatment are crucial for a good prognosis.'
    },
    'nv': {
        'full_name': 'Melanocytic Nevi',
        'causes': 'Commonly caused by genetic factors and sun exposure.',
        'next_steps': 'Typically benign and require no treatment. Monitor for changes in size, shape, or color and consult a dermatologist if changes occur.',
        'info': 'These are commonly known as moles. They are usually harmless but can sometimes develop into melanoma.'
    },
    'vasc': {
        'full_name': 'Vascular Lesions',
        'causes': 'Caused by abnormalities in blood vessels or lymph vessels. Can be congenital or acquired.',
        'next_steps': 'Consult a dermatologist or vascular specialist for assessment. Treatment options vary depending on the type of lesion and may include laser therapy or surgical removal.',
        'info': 'Vascular lesions include hemangiomas, port-wine stains, and spider veins. They are generally benign but can sometimes cause cosmetic concerns or complications.'
    }
}

# Function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    return img_array

# Function to predict the class of an image
def predict_image_class(model, img_path, class_names):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_idx]
    return predicted_class_name

# Streamlit app
st.title("Skin Disease Prediction")
st.write("Upload an image of a skin lesion and get a prediction of the disease.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Save the uploaded image to a temporary file
    img_path = "temp_image.jpg"
    img.save(img_path)

    # Add a progress bar
    progress_bar = st.progress(0)

    # Simulate processing time with progress bar updates
    for i in range(100):
        time.sleep(0.02)  # Simulate a delay in processing
        progress_bar.progress(i + 1)

    # Predict the class of the image
    predicted_class = predict_image_class(loaded_model, img_path, class_names)

    # Clear the progress bar once prediction is done
    progress_bar.empty()

    # Display the prediction
    st.write(f"**Predicted Disease:** {predicted_class}")
    st.write(f"**Full Name:** {class_descriptions[predicted_class]['full_name']}")

    # Display additional information about the disease
    st.write(f"**Information:** {class_descriptions[predicted_class]['info']}")
    st.write(f"**Causes:** {class_descriptions[predicted_class]['causes']}")
    st.write(f"**Next Steps:** {class_descriptions[predicted_class]['next_steps']}")
