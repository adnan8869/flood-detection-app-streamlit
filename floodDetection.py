import os
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Flood Detection System",
                   layout="wide",
                   page_icon="⛈️")

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to saved models inside 'save model' subfolder
rainfall_model_path = os.path.join(working_dir, 'save model', 'flood_detection_model.keras')
image_model_path = os.path.join(working_dir, 'save model', 'fine_tuned_flood_detection_model')

# Custom objects for loading models with compatibility issues
custom_objects = {
    'InputLayer': tf.keras.layers.InputLayer
}

# Load the rainfall-based model with error handling
@st.cache_resource
def load_rainfall_model():
    try:
        st.write(f"Attempting to load rainfall model from: {rainfall_model_path}")
        
        # Try different loading methods
        try:
            # Method 1: Direct loading with custom objects
            model = keras.models.load_model(rainfall_model_path, 
                                          custom_objects=custom_objects,
                                          compile=False)
            st.success("Rainfall-based model loaded successfully.")
            return model
        except Exception as e1:
            st.warning(f"Direct loading failed: {e1}")
            
            # Method 2: Try with different TensorFlow loading options
            try:
                with tf.keras.utils.custom_object_scope(custom_objects):
                    model = tf.keras.models.load_model(rainfall_model_path, compile=False)
                st.success("Rainfall-based model loaded successfully (method 2).")
                return model
            except Exception as e2:
                st.warning(f"Second method failed: {e2}")
                
                # Method 3: Try loading with SavedModel format
                try:
                    model = tf.saved_model.load(rainfall_model_path)
                    st.success("Rainfall-based model loaded as SavedModel.")
                    return model
                except Exception as e3:
                    st.error(f"All loading methods failed. Last error: {e3}")
                    return None
                    
    except FileNotFoundError:
        st.error(f"Rainfall model file not found at {rainfall_model_path}. Please ensure the file exists in the 'save model' subfolder.")
        return None
    except Exception as e:
        st.error(f"Error loading rainfall model: {e}")
        return None

# Load the image-based model with error handling
@st.cache_resource
def load_image_model():
    try:
        st.write(f"Attempting to load image model from: {image_model_path}")
        
        # Try different file extensions and loading methods
        possible_extensions = ['', '.h5', '.keras']
        
        for ext in possible_extensions:
            model_path = image_model_path + ext
            if os.path.exists(model_path):
                try:
                    # Method 1: Direct loading with custom objects
                    model = tf.keras.models.load_model(model_path, 
                                                     custom_objects=custom_objects,
                                                     compile=False)
                    st.success(f"Image-based model loaded successfully from {model_path}.")
                    return model
                except Exception as e1:
                    st.warning(f"Direct loading failed for {model_path}: {e1}")
                    
                    # Method 2: Try with custom object scope
                    try:
                        with tf.keras.utils.custom_object_scope(custom_objects):
                            model = tf.keras.models.load_model(model_path, compile=False)
                        st.success(f"Image-based model loaded successfully (method 2) from {model_path}.")
                        return model
                    except Exception as e2:
                        st.warning(f"Second method failed for {model_path}: {e2}")
                        continue
        
        # If no file found with any extension
        raise FileNotFoundError("No model file found with supported extensions")
        
    except FileNotFoundError:
        st.error(f"Image model file not found at {image_model_path}. Please ensure the file exists in the 'save model' subfolder with a supported extension (.h5 or .keras).")
        return None
    except Exception as e:
        st.error(f"Error loading image model: {e}")
        return None

# Load models
rainfall_model = load_rainfall_model()
image_model = load_image_model()

# Function to predict flood risk based on rainfall data
def predict_flood_risk(jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec):
    if rainfall_model is None:
        return "Error: Rainfall model not loaded."
    try:
        # Prepare input data
        input_data = np.array([[jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]])
        
        # Handle different model types
        if hasattr(rainfall_model, 'predict'):
            # Standard Keras model
            prediction = rainfall_model.predict(input_data, verbose=0)
            
            # Handle different prediction formats
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                # Multi-class output
                pred_class = np.argmax(prediction[0])
                probability = np.max(prediction[0])
            else:
                # Binary classification
                pred_class = 1 if prediction[0] > 0.5 else 0
                probability = float(prediction[0]) if prediction[0] <= 1 else float(prediction[0]) / 100
            
            result = "High risk of flood" if pred_class == 1 else "Low risk of flood"
            result += f" (Confidence: {probability:.2%})"
            
        else:
            # SavedModel format
            try:
                prediction = rainfall_model(input_data)
                pred_class = 1 if prediction.numpy()[0] > 0.5 else 0
                probability = float(prediction.numpy()[0])
                result = "High risk of flood" if pred_class == 1 else "Low risk of flood"
                result += f" (Confidence: {probability:.2%})"
            except:
                result = "Error: Could not make prediction with SavedModel format"
        
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

        # Handle different model types
        if hasattr(image_model, 'predict'):
            # Standard Keras model
            prediction = image_model.predict(img_array, verbose=0)
            
            # Handle different prediction formats
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                # Multi-class output
                probability = np.max(prediction[0])
                pred_class = np.argmax(prediction[0])
            else:
                # Binary classification
                probability = float(prediction[0][0]) if hasattr(prediction[0], '__len__') else float(prediction[0])
                pred_class = 1 if probability >= 0.5 else 0
            
        else:
            # SavedModel format
            try:
                prediction = image_model(img_array)
                probability = float(prediction.numpy()[0][0])
                pred_class = 1 if probability >= 0.5 else 0
            except:
                return "Error: Could not make prediction with SavedModel format"

        if pred_class == 1:
            return f"Prediction: High Flood Risk (Confidence: {probability:.2%})"
        else:
            return f"Prediction: Low Flood Risk (Confidence: {(1-probability):.2%})"
            
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
    st.title('🌧️ Rainfall-Based Flood Detection')

    # Input fields for monthly rainfall
    st.subheader("Enter Monthly Rainfall (mm)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        jan = st.number_input('January', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        may = st.number_input('May', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        sep = st.number_input('September', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
    with col2:
        feb = st.number_input('February', min_value=0.0, max_value=2000.0, step=0.1, value=0.0)
        jun = st.number_input('June', min_value=0.0, max_value=2000.0, step=0.1, value=0.1)
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
    if st.button('🔍 Run Rainfall-Based Detection'):
        # Validate inputs
        inputs = [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]
        if sum(inputs) == 0:
            st.warning("Please enter at least some rainfall data for accurate prediction.")
        elif any(val > 2000.0 for val in inputs):
            st.error("Rainfall values exceed the realistic upper limit (2000 mm max).")
        else:
            with st.spinner('Analyzing rainfall data...'):
                result = predict_flood_risk(jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec)
            
            if "High risk" in result:
                st.error(f"⚠️ **Result**: {result}")
            else:
                st.success(f"✅ **Result**: {result}")

# Image-Based Detection Section
if selected == 'Image-Based Detection':
    st.title('📸 Image-Based Flood Detection')
    st.subheader("Upload an Image for Flood Detection")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image (jpg or png)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
        
        with col2:
            # Run prediction
            if st.button('🔍 Run Image-Based Detection'):
                with st.spinner('Analyzing image...'):
                    result = predict_flood_image(uploaded_file)
                
                if "High Flood Risk" in result:
                    st.error(f"⚠️ **Result**: {result}")
                else:
                    st.success(f"✅ **Result**: {result}")

# Footer
st.markdown("---")
st.markdown("**Note**: This is a prototype flood detection system. For actual emergency situations, please contact local authorities.")

# Debug information (only shown if models failed to load)
if rainfall_model is None or image_model is None:
    with st.expander("Debug Information"):
        st.write(f"Python version: {os.sys.version}")
        st.write(f"TensorFlow version: {tf.__version__}")
        st.write(f"Working directory: {working_dir}")
        st.write(f"Rainfall model path: {rainfall_model_path}")
        st.write(f"Image model path: {image_model_path}")
        st.write(f"Rainfall model exists: {os.path.exists(rainfall_model_path)}")
        st.write(f"Image model exists: {os.path.exists(image_model_path)}")
        
        # List all files in the save model directory
        save_model_dir = os.path.join(working_dir, 'save model')
        if os.path.exists(save_model_dir):
            st.write(f"Files in save model directory: {os.listdir(save_model_dir)}")
        else:
            st.write("Save model directory does not exist")