import streamlit as st
import seaborn as sns
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)
cm_img = Image.open('confusion_matrix.jpg')

st.title("Tomato Sense")
st.write("---")
st.subheader("DASHBOARD")
st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.header("Confusion Matrix")

    st.image(cm_img)

with col2:
    st.header("Test Accuracy")
    st.header("89.23%")
    st.write("---")
    st.header("F1 Score")
    st.header("94.82%")
