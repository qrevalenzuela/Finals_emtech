import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
model = tf.keras.models.load_model('C:\\Users\\valen\\Downloads\\BEST_MODEL_STRONG.h5')

def predict(image)
    processed_image = preprocess_image(image)
   
    prediction = model.predict(processed_image)
  
    return prediction
def preprocess_image(image):
  
    image = image.resize((64, 64))
    

    image_array = np.array(image)
    
  
    image_array = image_array / 255.0
    
   
    processed_image = np.expand_dims(image_array, axis=0)
    
    return processed_image
def main():
    st.title("Multi-class Weather Prediction By Robin E. Valenzuela")
    st.write("Upload an image and we shall predict.")

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            with st.spinner('Predicting...'):
                prediction = predict(image)

            # Display the prediction
            class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
            class_index = np.argmax(prediction)
            class_name = class_names[class_index]

            st.success(f"Predicted Weather: {class_name}")

if __name__ == '__main__':
    main()

