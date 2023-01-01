import streamlit as st
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt
import random
from trace_model import inference, load_model

st.title("Tomato Sense")
st.write("---")
st.subheader("Test Our Model")
st.write("---")

user_img = st.file_uploader("Upload your image")
model = load_model('model.pt')

if user_img:
    user_img_file = Image.open(user_img)
    user_img_data = np.asarray(user_img_file)
    
    name = user_img.name
    # Extract the red and NIR bands from the image data
    red = user_img_data[:,:,0]
    nir = user_img_data[:,:,2]

    # Calculate the NDVI values for each pixel
    ndvi = (nir - red) / (nir + red)
    cm = plt.get_cmap('RdYlGn')
    colored_image = cm(ndvi)
    heatmap_img = Image.fromarray((colored_image * 255).astype(np.uint8))

    class_dis = inference(model, user_img)
    with st.spinner('Wait for it...'):
        time.sleep(5)
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("HEATMAP")
        st.image(heatmap_img)

    with col2:
        st.write("PREDICTION")
        st.header(class_dis.upper())

    
    
    

