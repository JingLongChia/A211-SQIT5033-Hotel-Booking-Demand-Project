import pandas as pd
import streamlit as st
import plotly.express as px

def app():
    st.title("Data Visualization 📈")
    st.write("### Table of data set")
    df = pd.read_csv('hotel_bookings.csv')
    st.dataframe(df)
    groupby_column = st.selectbox(
        'What would you like to analyse?',
        ('hotel','market_segment','deposit_type','customer_type','is_repeated_guest','booking_changes'),
    )

    ## Group DataFrame
    output_columns = ['is_canceled']
    df_grouped = df.groupby(by=[groupby_column],as_index=False)[output_columns].sum()

    fig = px.bar(
        df_grouped,
        x = groupby_column,
        y = 'is_canceled',
        title = f'<b>is_canceled Vs {groupby_column}</b>'
    )
    st.write("Bar Chart")
    st.plotly_chart(fig)
    
