import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

from streamlit_option_menu import option_menu

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title = "A Self-explaining and interactive tomato doctor",
  layout="wide", initial_sidebar_state="expanded")

col1,col2 = st.columns([1,5])
col3 = st.columns([0.5,5,2])
#col = st.columns([1])
col = st.columns([1])

#Image
from PIL import Image
img = Image.open("tomato2.png")
with st.container():
  col1.image(img, width = 200)
  col2.write("             ")
  col2.write("             ")
  XTomDoc = '<p Result: style="font-family:Bradley Hand ITC; color:Blue; font-size: 50px;"> <b>X-TOMDOC</b></u></p>'
  col2.markdown(XTomDoc, unsafe_allow_html=True)
col[0].write(":heavy_minus_sign:" * 43)

col[0].subheader("Welcome to X-TomDoc")
st.session_state.uploaded_image = None


