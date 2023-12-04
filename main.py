import streamlit as st
import numpy as np
import cv2
import predictor


predictor.load_model()

st.set_page_config(page_title='Celebrity Recognizer', page_icon='👑', layout='wide')
st.title('Celebrity Recognizer')
st.subheader('Upload a picture of a celebrity and see if the model can recognize them!')

hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''
st.markdown(hide_img_fs, unsafe_allow_html=True)

st.markdown('<h4>Here are the celebrities the model can recognize:</h4>',
            unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

col1.image('Leonardo DiCaprio.jpeg', width=300, caption='Leonardo DiCaprio')
col2.image('Johnny Depp.jpeg', width=300, caption='Johnny Depp')
col3.image('Natalie Portman.jpeg', width=300, caption='Natalie Portman')
col4.image('Kate Winslet.jpeg', width=300, caption='Kate Winslet')
col5.image('Megan Fox.jpeg', width=300, caption='Megan Fox')

st.markdown('<h5>You can drag and drop the above images to the upload area below to test the model!</h5>',
            unsafe_allow_html=True)

st.divider()

file = st.file_uploader('Upload a picture of a celebrity', type=['jpg', 'jpeg', 'png'])
if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    prediction = predictor.predict(image)
    if prediction['class'] == 'No face detected':
        st.error('No face detected')
    else:
        prediction_class = prediction['class']
        probability = prediction['class_probability']
        st.info(f'Prediction: {prediction_class}\n\nAccuracy: {probability}%')
