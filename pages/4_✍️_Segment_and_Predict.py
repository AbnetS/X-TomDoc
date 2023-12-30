import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import cv2
from PIL import Image

import utilities

st.set_page_config(layout="wide", page_title = "A Self-explaining and interactive tomato doctor")

c1,c2 = st.columns([1,1])
c3 = st.columns([1])
c4,c42,c5,c6,c7 = st.columns([3,0.5,3,0.5,3])
c8 = st.columns([1])
c9,c92,c10,c11 = st.columns([1.8,0.3,3,1])
c12 = st.columns([1])

choosen_model = c2.selectbox("Select model", ["Vgg16", "Vgg19","IncepV3", "SVgg16", 
                        "SVgg19","SIncepV3","SNVgg16", "SNVgg19","SNIncepV3"],index = 0)
st.session_state.choosen_model = choosen_model
st.session_state.model = utilities.load_model(choosen_model+".h5")


uploaded_file = c3[0].file_uploader("Press the button to upload tomato leaf image", type=["jpg","png","jpeg"])

for i in range(0,8):
    c6.text("=")
for i in range(0,8):
    c42.text("=")
for i in range(0,8):
    c92.text("=")

with st.container():
    if (uploaded_file is not None): 
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)   
        img = Image.open(uploaded_file)
        c4.image(uploaded_file, width=250) 
        #st.session_state.uploaded_image = uploaded_file  

        #Upload the same image the user was working with on another page
        #if (st.session_state.uploaded_image is not None):
            #c4.image(st.session_state.uploaded_image, width=250) 
    
        with st.container():  
            green_saturation = c5.selectbox("Green color intensity:",["10","20","30","40"],2)  
            brown_saturation = c5.selectbox("Brown color intensity:",["50","75","100","125","150", "200","255"],1)   
            bkgrnd_color = c5.selectbox("Background color:", ["Black","Red"],0)
            run_segment = True
        with st.container():            
            segment_clicked = c5.button("Segment and Predict", use_container_width=True)
            if (segment_clicked or run_segment):                
                index = 0
                if (uploaded_file is not None):
                    segmented = utilities.segment_leaves(file_bytes, None,green_saturation, brown_saturation, bkgrnd_color)
                else:
                    segmented= utilities.segment_leaves(None, st.session_state.uploaded_image,green_saturation, brown_saturation, bkgrnd_color)

                c7.image(segmented, width=250)
                model = utilities.load_model(choosen_model+".h5")
                if (choosen_model.__contains__("Vgg")):
                    input_arr = cv2.resize(np.array(segmented), (224,224))
                elif (choosen_model.__contains__("Incep")):
                    input_arr = cv2.resize(np.array(segmented), (299,299)) 

                
                input_arr = np.array([input_arr])  # Convert single image to a batch.
                input_arr = input_arr.astype('float32') / 255.
                run_predict = True               
                
                prediction = utilities.make_prediction(input_arr)
                max, index = utilities.returnMax(prediction)
                if (index == 0): 
                    result_text = '<p Result: style="font-family:Bahnschrift; color:Blue; font-size: 18px;">Result: <u> <b>Diseased: (Late blight)</b></u></p>'
                    c9.markdown(result_text, unsafe_allow_html=True)
                elif (index == 1): 
                    result_text = '<p Result: style="font-family:Bahnschrift; color:Blue; font-size: 18px;">Result: <u> <b>Diseased: (Septoria leaf spot)</b></u></p>'
                    c9.markdown(result_text, unsafe_allow_html=True)
                elif (index == 2): 
                    result_text = '<p style="font-family:Bahnschrift; color:Blue; font-size: 18px;">Result: <u> <b>Healthy</b></u></p>'
                    c9.markdown(result_text, unsafe_allow_html=True)

                #Change to percentage
                prediction[0][0] = prediction[0][0]*100
                prediction[0][1] = prediction[0][1]*100
                prediction[0][2] = prediction[0][2]*100
                
                c9.text("=================================")
                c9.text("Late blight = " + str(prediction[0][0]))
                c9.text("Septoria leaf spot = " + str(prediction[0][1]))
                c9.text("Healthy = " + str(prediction[0][2]))
                c9.text("=================================") 

                c8[0].text("_______________________________________________________________________________________________________________________")

                st.session_state.run_explain = True
                with st.container():  
                    explain_clicked = c10.button("Explain", key="explain_original_prediction",use_container_width=True)  
                    num_samples = c11.text_input("Samples:", value=200)
                    setting_text = '<p Result: style="font-family:Bahnschrift; color:Blue; font-size: 18px;"><u> <b>Settings</b></u></p>'
                    c11.text(" ")
                    c11.text(" ")
                    c11.text(" ")
                    c11.text(" ")
                    c11.text(" ")
                    c11.text(" ")
                    c11.text(" ")
                    c11.text(" ")
                    c11.text(" ")
                    #c11.markdown(setting_text, unsafe_allow_html=True)
                    c11.text("=================================")
                    top_segments = c11.selectbox("Top regions", [1,2,3,4,5,6,7,8,9,10])
                    
                    hide_image = c11.checkbox("Hide Image", value = False) 
                    run_explained = True
                    c11.text(" ")  
                    c11.text(" ") 
                    c11.text(" ") 
                    c11.text("=================================")  
                    scroll_text = '<p Result: style="font-family:Bahnschrift; color:Red; font-size: 14px;"> <b>Scroll down to see the contribution of all segmented regions. </b></p>'
                    c11.markdown(scroll_text, unsafe_allow_html=True)
                    explanation_text = '<p Result: style="font-family:Bahnschrift; color:Blue; font-size: 18px;"><u> <b>Explanation</b></u></p>'
                    if (explain_clicked or st.session_state.run_explain==True):
                        run_segment = True
                        explanation_text = '<p Result: style="font-family:Bahnschrift; color:Blue; font-size: 18px;"><u> <b>Explanation</b></u></p>'
                        #c5.markdown(explanation_text, unsafe_allow_html=True)           
                        choice = c10.selectbox("Choose result to explain", ["Late blight", "Septoria leaf spot", "Healthy"],index = index)
                        index = choice
                        st.session_state.run_explain = True
                        run_explained = True
                        placeholder = c10.empty()
                        with placeholder, st.spinner("In progress..."):
                            c10.text("Explaining " + choice)
                            label  = utilities.change_class_to_label(choice)  
                            explanation = utilities.explain_instance(input_arr, int(num_samples),0.25, model)
                            features = explanation.local_exp[label]
                            sorted_features = sorted(features, key=lambda x:x[1], reverse=True)
                            print(sorted_features)
                            
                            explanation.local_exp[label] = sorted_features[0:int(top_segments)]                  
                            img,mask = utilities.get_explanation(explanation, label, hide_image)         
                            utilities.plot_comparison(segmented, img, mask,c10)           

                            print(explanation.local_pred)
                            print(explanation.score)
                            confidence = ((explanation.score[0]+explanation.score[1]+explanation.score[2])/3)*100 
                            confidence_str = "Confidence=" + str(round(confidence,2)) + "%"
                            confidence_text = f'<p Result: style="font-family:Bahnschrift; color:Black; font-size: 14px;"> <b>{confidence_str}</b></p>'          
                            
                            #c11.text("Confidence=" + str(round(confidence,2)) + "%")
                            c11.markdown(confidence_text, unsafe_allow_html=True)
                                    
                            #Map each explanation weight to the corresponding superpixel
                            explanation.local_exp[label] = features
                            dict_heatmap = dict(explanation.local_exp[label])            
                            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)             
                            utilities.plot_detailed_explanation(img, explanation,heatmap, c12[0])
                            #c10.text("Explaining " + choice)
                            #label  = utilities.change_class_to_label(choice)            
                            #img,mask = utilities.explain_instance(input_arr, label, int(num_samples),choosen_model)
                            #utilities.plot_comparison(segmented, img, mask,c10)





