import os
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow import keras


# Set page configuration
st.set_page_config(page_title="Flood Detection System",
                   layout="wide",
                   page_icon="⛈️")

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to saved models inside 'save model' subfolder
rainfall_model_path = os.path.join(working_dir, 'save model', 'flood_detection_model.keras')
image_model_path = os.path.join(working_dir, 'save model', 'fine_tuned_flood_detection_model')

# Load the rainfall-based model
try:
    st.write(f"Attempting to load rainfall model from: {rainfall_model_path}")
    rainfall_model = keras.models.load_model(rainfall_model_path, compile=False)
    st.success("Rainfall-based model loaded successfully.")

except FileNotFoundError:
    st.error(f"Rainfall model file not found at {rainfall_model_path}. Please ensure the file exists in the 'save model' subfolder.")
    rainfall_model = None
except Exception as e:
    st.error(f"Error loading rainfall model: {e}")
    rainfall_model = None

# Load the image-based model
try:
    st.write(f"Attempting to load image model from: {image_model_path}")
    possible_extensions = ['', '.h5', '.keras']
    for ext in possible_extensions:
        model_path = image_model_path + ext
        if os.path.exists(model_path):
            image_model = tf.keras.models.load_model(model_path)
            st.success("Image-based model loaded successfully.")
            break
    else:
        raise FileNotFoundError
except FileNotFoundError:
    st.error(f"Image model file not found at {image_model_path}. Please ensure the file exists in the 'save model' subfolder with a supported extension (.h5 or .keras).")
    image_model = None
except Exception as e:
    st.error(f"Error loading image model: {e}")
    image_model = None

# Function to predict flood risk based on rainfall data
def predict_flood_risk(jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec):
    if rainfall_model is None:
        return "Error: Rainfall model not loaded."
    try:
        # Prepare input data
        input_data = np.array([[jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]])
        # Predict using the loaded model    
        prediction = rainfall_model.predict(input_data)
        probability = rainfall_model.predict_proba(input_data)[0][1] if hasattr(rainfall_model, 'predict_proba') else None
        result = "High risk of flood" if prediction[0] == 1 else "Low risk of flood"
        if probability is not None:
            result += f" (Probability: {probability:.2%})"
        return result
    except Exception as e:
        return f"Error in prediction: {e}"

# Function to preprocess and predict flood from image
def predict_flood_image(file):
    if image_model is None:
        return "Error: Image model not loaded."
    try:
        # Resize image to model's expected input size (100x100)
        img = image.load_img(file, target_size=(100, 100))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)       # Shape: (1, 100, 100, 3)
        img_array = img_array / 255.0                        # Normalize

        # Predict
        prediction = image_model.predict(img_array)
        probability = prediction[0][0]  # Assuming binary classification: flood probability

        if probability >= 0.5:
            return f"Prediction: High Flood Risk"
        else:
            return f"Prediction: Low Flood Risk"
    except Exception as e:
        return f"Error in image prediction: {e}"




# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Flood Detection System',
                           ['Rainfall-Based Detection', 'Image-Based Detection'],
                           menu_icon='warning',
                           icons=['cloud-rain', 'image'],
                           default_index=0)

# Rainfall-Based Detection Section
if selected == 'Rainfall-Based Detection':
    st.title('Rainfall-Based Flood Detection')

    # Input fields for monthly rainfall
    st.subheader("Enter Monthly Rainfall (mm)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        jan = st.number_input('January', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        may = st.number_input('May', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        sep = st.number_input('September', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
    with col2:
        feb = st.number_input('February', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        jun = st.number_input('June', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        oct = st.number_input('October', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
    with col3:
        mar = st.number_input('March', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        jul = st.number_input('July', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        nov = st.number_input('November', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
    with col4:
        apr = st.number_input('April', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        aug = st.number_input('August', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        dec = st.number_input('December', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)

    # Button to run rainfall-based detection
    if st.button('Run Rainfall-Based Detection'):
        # Validate inputs
        inputs = [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]
        if any(val <= 0.0 for val in inputs):
            st.error("All fields must be filled")
        elif any(val > 2000.0 for val in inputs):
            st.error("Rainfall values exceed the realistic upper limit (2000 mm max).")
        else:
            result = predict_flood_risk(jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec)
            st.write(f"**Result**: {result}")

# Image-Based Detection Section
if selected == 'Image-Based Detection':
    st.title('Image-Based Flood Detection')
    st.subheader("Upload an Image for Flood Detection")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image (jpg or png)", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        
        # Run prediction
        if st.button('Run Image-Based Detection'):
            result = predict_flood_image(uploaded_file)
            st.write(f"**Result**: {result}")

# Instructions to run the app
# python -m streamlit run FloodDetection.py