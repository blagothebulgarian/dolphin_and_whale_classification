import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import altair as alt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing import image



st.title('Love Dolphins & Whales?')
st.markdown('This super classifier here will classify your marine spotting photo as one of a total of 30 different whale and dolphin species!')


img_file_buffer = st.file_uploader("Choose a file")

if img_file_buffer is not None:


   

    bytes_data = img_file_buffer.getvalue()
    image = np.array(tf.io.decode_image(bytes_data, channels=3))
    #st.write(type(image))
    #st.write(image.shape)
    
    input_shape = (224, 224)
    image = tf.keras.preprocessing.image.smart_resize( image, input_shape, interpolation='bilinear')
    #st.write(type(image))
    #st.write(image.shape)

    #st.write(image[0])
    
    image = image/255
   # st.write(image[0])
    image = np.expand_dims(image, axis=0)
    #st.write(image.shape)
with CustomObjectScope(
    {'GlorotUniform': glorot_uniform()}):
    model = load_model('./models/mobilenet_transfer_model.h5')

with open("./prediction_dict.json", "r") as file:
    dictionary = json.load(file)

if st.button('Submit'):
    preds = model.predict(image)
    pred = dictionary[str(np.argmax(preds))]
    pred = ' '.join(pred.split('_')[1:])
    st.write(f"This is likely a picture of a {pred}!")
    
    
    preds = list(preds[0])
    
    data =[]
    for index, pred in enumerate(preds):
        categ = dictionary[str(index)]
        categ = ' '.join(categ.split('_')[1:])
        data.append([categ,pred])
    
    df = pd.DataFrame(data).rename(columns={0: "Species Class", 1: "Certainty Probability %"})
    df = df.sort_values(by=['Certainty Probability %'], ascending = False).head(15)

    bars = alt.Chart(df).mark_bar().encode(
    y=alt.Y('Species Class:N', sort='-x'),
    x="Certainty Probability %:Q",
    color=alt.Color('Species Class', scale=alt.Scale(scheme='blues'), legend = None)
    
)

    text = bars.mark_text(
    align='left',
    baseline='middle',
    dx=3  # Nudges text to right so it doesn't appear on top of the bar
).encode(
    text='Certainty Probability %:Q'
)

    chart = (bars + text).properties(height=900, title = 'Top 15 Class Probabilities (%)')
    st.altair_chart(chart, use_container_width=True)

st.subheader('Our 30 Species:')
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("- Beluga Whale")
    st.markdown("- Blue Whale")
    st.markdown("- Bottlenose Dolphin")
    st.markdown("- Bottlenose Dolpin")
    st.markdown("- Brydes Whale")
    st.markdown("- Commersons Dolphin")
    st.markdown("- Common Dolphin")
    st.markdown("- Cuviers Beaked Whale")
    st.markdown("- Dusky Dolphin")
    st.markdown("- False Killer Whale")

with col2:
    st.markdown("- Fin Whale")
    st.markdown("- Frasiers Dolphin")
    st.markdown("- Globis")
    st.markdown("- Gray Whale")
    st.markdown("- Humpback Whale")
    st.markdown("- Kiler Whale")
    st.markdown("- Killer Whale")
    st.markdown("- Long Finned Pilot Whale")
    st.markdown("- Melon Headed Whale")
    st.markdown("- Minke Whale")

with col3:
    st.markdown("- Pantropic Spotted Dolphin")
    st.markdown("- Pilot Whale")
    st.markdown("- Pygmy Killer Whale")
    st.markdown("- Rough Toothed Dolphin")
    st.markdown("- Sei Whale")
    st.markdown("- Short Finned Pilot Whale")
    st.markdown("- Southern Right Whale")
    st.markdown("- Spinner Dolphin")
    st.markdown("- Spotted Dolphin")
    st.markdown("- White Sided Dolphin")