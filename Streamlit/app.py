import pandas as pd
import streamlit as st
import Home
import Prediction
import Visualization

PAGES = {
    "Home": Home,
    "Data Visualization": Visualization,
    "Data Prediction": Prediction,
}

html3 = '''
<h2 style="text-align: center;font-size: 30px">A211 SQIT5033 Hotel Booking Demand Project</h2>
<hr class="rounded">
'''
st.sidebar.markdown(html3, unsafe_allow_html=True)

st.sidebar.title("Explore")

selection = st.sidebar.radio("", list(PAGES.keys()))
page = PAGES[selection]
page.app()

html4 = '''
<br>
<p><b>Data Visualization -</b> Analyze the hotel booking demand data accross various Exploratory data analysis using
bar chart.</p>
<br>
<p><b>Data Prediction -</b> Predict the possible canceled hotel booking using Decision Tree classfication algorithm.</p>
'''
st.sidebar.markdown(html4, unsafe_allow_html=True)
