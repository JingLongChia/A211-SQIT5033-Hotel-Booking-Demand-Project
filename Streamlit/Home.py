import pandas as pd
import streamlit as st

def app():

    st.title("Hotel Booking Demand Project 📝")
    URL = 'https://drive.google.com/file/d/1gmsEtYB2NQyJcc0pkB6jXtqijdHpGiEH/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+URL.split('/')[-2]
    st.image(path)

    st.write("### Data Description")
    st.write("We will use the Hotel Booking Demand dataset from the Kaggle.")
    st.write("You can download it from here: https://www.kaggle.com/jessemostipak/hotel-booking-demand")
    st.write("This data set contains booking information for a city hotel and a resort hotel and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things. All personally identifying information has from the data.")
