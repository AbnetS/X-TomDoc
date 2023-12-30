import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import cv2
import math

def load_model(model_name):
    st.session_state.model = keras.saving.load_model("models/" + model_name)
    

def make_prediction(img):
    if (st.session_state.model is None):
        load_model(st.session_state.choosen_model + ".h5")
    return st.session_state.model.predict(img) 

def returnMax(prediction):
  index = 0
  max = prediction[0][0]
  if (prediction[0][1]>max): 
    max =  prediction[0][1]
    index = 1
  if (prediction[0][2]>max): 
    max =  prediction[0][2]
    index = 2
  return max, index

def plot_comparison(main_image, img, mask, col):
    fig = plt.figure(figsize=(15,5))
    #ax = fig.add_subplot(141)
    #ax.imshow(main_image, cmap="gray")
    #ax.set_title("Original Image")    
    #ax = fig.add_subplot(142)
    #ax.imshow(img)
    #ax.set_title("Image")
    #ax = fig.add_subplot(142)
    #ax.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    #pcm = ax.pcolormesh(np.random.random((20, 20)) * (0 + 1), cmap='RdBu')
    #fig.colorbar(pcm)
    #ax.set_title("Mask")
    ax = fig.add_subplot(141)
    ax.imshow(mark_boundaries(img, mask, color = (1,1,0),mode = "thick"))    
    ax.set_title("Top contributing regions")
    ax.axes.title.set_fontsize(('small'))  
    ax.tick_params(axis='both', which='major', labelsize=6)
        
    col.pyplot(fig)
    
    col.text("__________________________________________________________________")

def plot_detailed_explanation(img, explanation, heatmap, col):
  fig = plt.figure(figsize=(15,5))
  ax = fig.add_subplot(141)  
  ax.imshow(mark_boundaries(img,explanation.segments,color=(1,1,0), mode = "thick"))  
  ax.set_title("Examined segments")
  ax.axes.title.set_fontsize(('small'))  
  ax.tick_params(axis='both', which='major', labelsize=6)
  ax =  fig.add_subplot(142)  
  im=ax.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())  
  ax.set_title("Heatmap") 
  ax.axes.title.set_fontsize(('small'))  
  ax.tick_params(axis='both', which='major', labelsize=6)
  fig.colorbar(im,ax=ax)
  col.pyplot(fig)

@st.cache_data
def explain_instance(img, num_samples, kernel_size,_model):
  print (str(kernel_size))
  explainer = lime_image.LimeImageExplainer(verbose=True, kernel_width=kernel_size)  
  explanation = explainer.explain_instance(img.squeeze(), make_prediction,num_samples = num_samples, distribution="comb_exp", progress_bar=True)
  print("score", explanation.score)
  print("pred", explanation.local_pred)
  return explanation

def get_explanation(explanation, class_label, hide_image):
  img2, mask = explanation.get_image_and_mask(class_label, positive_only=True, negative_only=False, 
                                        hide_rest=hide_image) 
  return img2,mask


def change_label_to_class(label):
  if (label == 0): return "Late blight"
  elif(label == 1): return "Septoria leaf spot"
  else: return "Healthy"

def change_class_to_label(class_name):
  if (class_name == "Late blight"): return 0
  elif(class_name == "Septoria leaf spot"): return 1
  else: return 2

def segment_leaves(file, image, green_saturation, brown_saturation, bkgrnd_color):

  leaf = cv2.imdecode(file, 1)
  leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB) 
  #else:
  #leaf = np.array(image)
  
  hsv_leaf = cv2.cvtColor(leaf, cv2.COLOR_RGB2HSV)
  light_green = (25,int(green_saturation),0) 
  dark_green = (100,255,255) 
  mask = cv2.inRange(hsv_leaf, light_green,dark_green)
  green_filtered = cv2.bitwise_and(leaf, leaf, mask=mask)
  light_brown = (5,20,21) 
  dark_brown = (80,int(brown_saturation),255) 
  mask_brown = cv2.inRange(hsv_leaf, light_brown,dark_brown)
  #to get part of leaves that indicate diseases since filtering green only may cut out those parts, however brown soils are not filtered out mostly
  brown_filtered = cv2.bitwise_and(leaf, leaf, mask=mask_brown)  
  final_mask = mask + mask_brown
  final_result = cv2.bitwise_and(leaf, leaf, mask=final_mask)
  if (bkgrnd_color == "Red"):
    black_pixels = np.where(
      (final_result[:, :, 0] == 0) & 
      (final_result[:, :, 1] == 0) & 
      (final_result[:, :, 2] == 0)
    )

    # set those pixels to red
    final_result[black_pixels] = [255, 0, 0]
  
  return final_result

def predict(image):
  model = load_model(st.session_state.choosen_model+".h5")  
  prediction = make_prediction(image)
  max, index = returnMax(prediction)  
  return index, prediction

