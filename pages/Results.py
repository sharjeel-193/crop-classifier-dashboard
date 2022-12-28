import streamlit as st
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

st.title("AgroVision")
st.write("---")
st.subheader("Test Our Model")
st.write("---")

user_img = st.file_uploader("Upload your image", type=['png', 'tif'])

if user_img:
    user_img = Image.open(user_img)
    user_img_data = np.asarray(user_img)
    # st.image(user_img_data, caption="Original", width=600, use_column_width=True, clamp=True)

    # Extract the red and NIR bands from the image data
    red = user_img_data[:,:,0]
    nir = user_img_data[:,:,3]

    # Calculate the NDVI values for each pixel
    ndvi = (nir - red) / (nir + red)
    cm = plt.get_cmap('RdYlGn')
    colored_image = cm(ndvi)
    heatmap_img = Image.fromarray((colored_image * 255).astype(np.uint8))

    st.image(heatmap_img, caption="NDVI Heatmap", width=600, use_column_width=True, clamp=True)