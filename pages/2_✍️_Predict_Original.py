import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import cv2
from PIL import Image
from keras.applications.vgg19 import preprocess_input
import math


import utilities

st.set_page_config(layout="wide", page_title = "A Self-explaining and interactive tomato doctor")

c1,c2 = st.columns([1,1])
c3 = st.columns([1])
c4,c42,c5,c52 = st.columns([1.8,0.3,3,1])
c6,c7 = st.columns([1.5,1])
c8 = st.columns([1])
choosen_model = c2.selectbox("Select model", ["Vgg16", "Vgg19","IncepV3", "SVgg16", 
                        "SVgg19","SIncepV3","SNVgg16", "SNVgg19","SNIncepV3"],index = 0)
st.session_state.choosen_model = choosen_model
st.session_state.model = utilities.load_model(choosen_model+".h5")

uploaded_file = c3[0].file_uploader("Press the button to upload tomato leaf image", type=["jpg","png","jpeg"])

for i in range(0,18):
    c42.text("=")

with st.container():
    #Upload image
    if (uploaded_file is not None): 
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)   
        img = Image.open(uploaded_file)
        c4.image(uploaded_file, width=250) 
        st.session_state.uploaded_image = uploaded_file    
                
        index = 0
        model = utilities.load_model(choosen_model+".h5") 
        if (choosen_model.__contains__("Vgg")):
            image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224))  
        elif (choosen_model.__contains__("Incep")):
            image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(299, 299))      
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        input_arr = input_arr.astype('float32') / 255.
        run_predict = True
                
        index, prediction = utilities.predict(input_arr)            
        if (index == 0): 
            result_text = '<p Result: style="font-family:Bahnschrift; color:Blue; font-size: 18px;">Result: <u> <b>Diseased: (Late blight)</b></u></p>'
            c4.markdown(result_text, unsafe_allow_html=True)
        elif (index == 1): 
            result_text = '<p Result: style="font-family:Bahnschrift; color:Blue; font-size: 18px;">Result: <u> <b>Diseased: (Septoria leaf spot)</b></u></p>'
            c4.markdown(result_text, unsafe_allow_html=True)
        elif (index == 2): 
            result_text = '<p style="font-family:Bahnschrift; color:Blue; font-size: 18px;">Result: <u> <b>Healthy</b></u></p>'
            c4.markdown(result_text, unsafe_allow_html=True)

        print (prediction)
        
        #Change to percentage
        prediction[0][0] = prediction[0][0]*100
        prediction[0][1] = prediction[0][1]*100
        prediction[0][2] = prediction[0][2]*100
        
        c4.text("=================================")
        c4.text("Late blight = " + str(prediction[0][0]))
        c4.text("Septoria leaf spot = " + str(prediction[0][1]))
        c4.text("Healthy = " + str(prediction[0][2]))
        c4.text("=================================") 

st.session_state.run_explain = True
with st.container():     
    if (c5.button("Explain", key="explain_original_prediction",use_container_width=True) or st.session_state.run_explain==True):
        num_samples = c52.text_input("Samples:", value=200)
        setting_text = '<p Result: style="font-family:Bahnschrift; color:Blue; font-size: 18px;"><u> <b>Settings</b></u></p>'
        c52.text(" ")
        c52.text(" ")
        c52.text(" ")
        c52.text(" ")
        c52.text(" ")
        c52.text(" ")
        c52.text(" ")
        c52.text(" ")
        c52.text(" ")
        #c52.markdown(setting_text, unsafe_allow_html=True)
        c52.text("=================================")
        top_segments = c52.selectbox("Top regions", [1,2,3,4,5,6,7,8,9,10])
        
        hide_image = c52.checkbox("Hide Image", value = False) 
        c52.text(" ")  
        c52.text(" ") 
        c52.text(" ") 
        c52.text("=================================")  
        scroll_text = '<p Result: style="font-family:Bahnschrift; color:Red; font-size: 14px;"> <b>Scroll down to see the contribution of all segmented regions. </b></p>'
        c52.markdown(scroll_text, unsafe_allow_html=True)
         
        run_predict = True
        explanation_text = '<p Result: style="font-family:Bahnschrift; color:Blue; font-size: 18px;"><u> <b>Explanation</b></u></p>'
        #c5.markdown(explanation_text, unsafe_allow_html=True)           
        choice = c5.selectbox("Choose result to explain", ["Late blight", "Septoria leaf spot", "Healthy"],index = index)
        index2 = choice
        st.session_state.run_explain = True
        placeholder = c5.empty()
        with placeholder, st.spinner("In progress..."):
            c5.text("Explaining " + choice)
            label  = utilities.change_class_to_label(choice)  
            explanation = utilities.explain_instance(input_arr, int(num_samples),0.25, model)
            features = explanation.local_exp[label]
            sorted_features = sorted(features, key=lambda x:x[1], reverse=True)
            print(sorted_features)
            
            explanation.local_exp[label] = sorted_features[0:int(top_segments)]                  
            img,mask = utilities.get_explanation(explanation, label, hide_image)         
            utilities.plot_comparison(image, img, mask,c5)           

            print(explanation.local_pred)
            print(explanation.score)
            confidence = ((explanation.score[0]+explanation.score[1]+explanation.score[2])/3)*100 
            confidence_str = "Confidence=" + str(round(confidence,2)) + "%"
            confidence_text = f'<p Result: style="font-family:Bahnschrift; color:Black; font-size: 14px;"> <b>{confidence_str}</b></p>'          
            
            #c52.text("Confidence=" + str(round(confidence,2)) + "%")
            c52.markdown(confidence_text, unsafe_allow_html=True)
                    
            #Map each explanation weight to the corresponding superpixel
            explanation.local_exp[label] = features
            dict_heatmap = dict(explanation.local_exp[label])            
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)             
            utilities.plot_detailed_explanation(img, explanation,heatmap, c8[0])
            


