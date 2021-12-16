import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix,classification_report, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
import joblib

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

## Streamlit Design
def app():
    st.title("Hotel Cancellation Prediction 📚")
    st.subheader("Predict if booking will be cancelled or confirmed using Decison Tree Algorithm")

    with st.form(key='myform'):
            option1 = st.selectbox('Hotel',
            ('City Hotel', 'Resort Hotel',))
            option2 = st.number_input("Lead Time", min_value=0, step=1)
            option3 = st.selectbox('Market Segment',
            ('Aviation', 'Complementary','Corporate','Direct','Groups','Offline TA/TO','Online TA',))
            option4 = st.selectbox('Deposit Type',
            ('No Deposit', 'Non Refund','Refundable',))
            option5 = st.selectbox('Customer Type',
            ('Contract', 'Group','Transient','Transient-Party',))
            option6 = st.selectbox('Reserved Room as before',
            ('Yes', 'No',))
            option7 = st.selectbox('Canceled More Than Bookings',
            ('Yes', 'No',))
            submit_button = st.form_submit_button('Predict')

            answer1 = option1
            answer2 = option2
            answer3 = option3
            answer4 = option4
            answer5 = option5
            answer6 = option6
            if answer6 == 'Yes':
                answer6 = 1
            else:
                answer6 = 0
            answer7 = option7
            if answer7 == 'Yes':
                answer7 = 1
            else:
                answer7 = 0

## Read data

    df = pd.read_csv('hotel_bookings.csv')

    df_subset = df.copy()
    df_subset['Room'] = 0
    df_subset.loc[ df_subset['reserved_room_type'] == df_subset['assigned_room_type'] , 'Room'] = 1
    df_subset['net_cancelled'] = 0
    df_subset.loc[ df_subset['previous_cancellations'] > df_subset['previous_bookings_not_canceled'] , 'net_cancelled'] = 1
    df_subset = df_subset.drop(['arrival_date_year','arrival_date_week_number','arrival_date_day_of_month',
                                'arrival_date_month','assigned_room_type','reserved_room_type','reservation_status_date',
                                'previous_cancellations','previous_bookings_not_canceled'],axis=1)
    df_subset = df_subset.drop(['reservation_status'], axis=1)
    df = df_subset

    ## Predict model
    ## Train and Test split on 80 percent and 20 percent.
    X = df[['hotel','lead_time', 'market_segment', 'deposit_type', 'customer_type','Room','net_cancelled']]
    y = df['is_canceled']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 42)

    cat_columns = ['hotel','market_segment','deposit_type','customer_type']
    num_columns = ['lead_time','Room','net_cancelled']

    ## pipeline model to predict all using Decision Tree
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder())
    ])

    numerical_pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])

    prepocessor = ColumnTransformer([
        ('categorical',categorical_pipeline,cat_columns),
        ('numerical', numerical_pipeline,num_columns)
    ])

    pipe_DT = Pipeline([
        ("prep", prepocessor),
        ("algo", DecisionTreeClassifier())
    ])

    ## Decision Tree model
    pipe_DT.fit(X_train, y_train)
    y_pred_DT_base =  pipe_DT.predict(X_test)
    y_pred_DT_base_train = pipe_DT.predict(X_train)

    recall_DT_base = recall_score(y_test, y_pred_DT_base)
    acc_DT_base = accuracy_score(y_test, y_pred_DT_base)
    precision_DT_base = precision_score(y_test, y_pred_DT_base)
    f1_DT_base = f1_score(y_test, y_pred_DT_base)
    acc_DT_base_train = accuracy_score(y_train, y_pred_DT_base_train)

    ## Import data
    data_user = {
        "hotel" : answer1,
        "lead_time" : answer2,
        "market_segment" : answer3,
        "deposit_type" : answer4,
        'customer_type': answer5,
        'Room':answer6,
        'net_cancelled':answer7,
    }

    user = pd.DataFrame(data = data_user, index = [1])
    pipe_DT.predict_proba(user)
    Predict = pipe_DT.predict_proba(user)

    ## Predict answer
    if submit_button:
        st.write("Probability Cancel Booking")
        df = pd.DataFrame(Predict,columns = ['Canceled','Confirmed'],index =["Probability"])
        st.write(df)
