import pandas as pd
import streamlit as st
import plotly.express as px

pd.options.display.max_columns = None

def app():
    st.title("Data Visualization 📈")
    st.write("### Table of data set")
    df = pd.read_csv('hotel_bookings.csv')

    ## Deal Missing Data
    df[['agent','company']] = df[['agent','company']].fillna(0.0)
    df['country'].fillna(df.country.mode().to_string(), inplace=True)
    df['children'].fillna(round(df.children.mean()), inplace=True)
    df = df.drop(df[(df.adults+df.babies+df.children)==0].index)
    df[['children', 'company', 'agent']] = df[['children', 'company', 'agent']].astype('int64')

    st.dataframe(df)

    ## Bar Chart
    st.write("## Bar Chart")
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
    st.plotly_chart(fig)
