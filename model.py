import streamlit as st
import tf_keras
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow_hub as hub

# Load the pre-trained model
model = tf_keras.models.load_model('flower_model.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# Define the Streamlit app
def main():
    st.title("Flower Classification App")
    st.write("Aren't familiar with the different types of flowers? Let me help you identify them!")
    st.write("This app can classify images of roses, daisies, dandelions, sunflowers, and tulips.")
    
    # Upload image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        prepped = preprocess_image(image)


        # Make predictions
        predictions = model.predict(prepped)
        predicted_class = get_predicted_class(predictions)

        # Display the predicted class
        st.success('This flower is most likely a {}'.format(predicted_class))

# Preprocess the image
def preprocess_image(image):
    # Convert the image to a numpy array
    image_array = np.array(image)
    #Standardise the pixel values
    image_array = image_array / 255.0
    # Resize the image to match the input size of the model
    resized_image = image_array.resize(224,224)
    # Expand the dimensions to match the input shape of the model
    expanded_image = resized_image[np.newaxis,]
    # Preprocess the image (e.g., normalize pixel values)
    preprocessed_image = tf_keras.applications.mobilenet_v2.preprocess_input(expanded_image)
    return preprocessed_image

# Get the predicted class
def get_predicted_class(predictions):
    # Get the index with the highest probability
    predicted_index = np.argmax(predictions)
    # Map the index to the corresponding class label
    class_labels = ['Rose', 'Daisy', 'Dandelion', 'Sunflower', 'Tulip']  # Replace with your own class labels
    predicted_class = class_labels[predicted_index]
    return predicted_class

if __name__ == '__main__':
    main()