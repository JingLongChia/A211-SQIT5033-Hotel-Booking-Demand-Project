# A211 SQIT5033 Hotel Booking Demand Project

![alt text](https://www.hotelspeak.com/wp-content/uploads/2019/05/hotel-direct-booking-strategy.jpg)

## Introduction to Dataset

We will use the Hotel Booking Demand dataset from the Kaggle.

You can download it from here: https://www.kaggle.com/jessemostipak/hotel-booking-demand

This data set contains booking information for a city hotel and a resort hotel and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things. All personally identifying information has from the data.


## Problem Statement

With the increase trend of cancellation from year to year, some hotel have think that high cancellation in hotel is the new norm of the industry which is a completely wrong approach, one out of four hotel guests are cancelling hotel booking ahead of a stay. This cancellation trend has effect the hotel not being able to accurately forecast occupancy within their revenue management, and the trend of cancellation also have causes hotel loss in opportunity cost (unsold room due to cancellation).


## Hypothesis

1. There is no relationship between arrival_ months and is_canceled.

2. There is no relationship between arrival_date_ year and is_canceled.

3. There is no relationshio between arrival_date_month and is_canceld.


## Goals

1. The Goals of this project is to find out the characteristic of customers who cancelled and finding a pattern in cancelled booking by doing an exploratory data analysis.

2. Building descriptive data mining to determine the data regularities and predictive data mining to predict cancellation.

3. Build and Deploy web application / dashboard using Streamlit from predictive data mining, that can predict of cancellation based on user input.


## Exploratory data analysis



## Data pre-processing

### 1. Dealing with Missing Values

Check if our data contains any missing values

```Python
df = df.copy()
df.isnull().sum().sort_values(ascending=False)[:10]
```
![image](https://user-images.githubusercontent.com/92434335/146466360-bd188cf8-2005-446e-a2e3-1a125c4c95ad.png)

We have 4 features with missing values.
In the agent and the company column, we have id_number for each agent or company, so for all the missing values, we will just replace it with 0.

```Python
## If no id of agent or company is null, just replace it with 0
df[['agent','company']] = df[['agent','company']].fillna(0.0)
```

Children column contains the count of children, so we will replace all the missing values with the rounded mean value.
And our country column contains country codes representing different countries. It is a categorical feature so I will also replace it with the mode value. The mode value is the value that appears more than any other value. So, in this case, I am replacing it with the country that appears the most often.

```Python
## For the missing values in the country column, replace it with mode (value that appears most often)
df['country'].fillna(data.country.mode().to_string(), inplace=True)


## for missing children value, replace it with rounded mean value
df['children'].fillna(round(data.children.mean()), inplace=True)
```
There are many rows that have zero guests including adults, children and babies. We will just remove these rows.

```Python
## Drop Rows where there is no adult, baby and child
df = df.drop(df[(df.adults+df.babies+df.children)==0].index)
```
### 2. Converting Datatype

Let’s check the datatype of each column in our dataset.

```Python
df.dtypes
```
![image](https://user-images.githubusercontent.com/92434335/146466790-3ac0955e-8332-470e-8965-1c1727f6072f.png)

We can see different data types for different columns.
There are some columns like children, company, and agent, that are float type but their values are only in integers.

```Python
## convert datatype of these columns from float to integer
df[['children', 'company', 'agent']] = df[['children', 'company', 'agent']].astype('int64')
```
So we will convert them to the integer type.

## Descriptive and predictive data mining solution

### 1. Descriptive data mining

### 2. Predictive data mining

- Feature Selection

  - Create more relevant features and remove irrelevant or less important features.
  - New feature call it Room which will contain 1 if the guest was assigned the same room that was reserved else 0.
  - Another feature will be net_cancelled will contain 1 If the current customer has canceled more bookings in the past than the number of bookings he did not cancel, else 0.
  - Now remove these unnecessary features : 'arrival_date_year','arrival_date_week_number','arrival_date_day_of_month',
                                            'arrival_date_month','assigned_room_type','reserved_room_type','reservation_status_date',
                                            'previous_cancellations','previous_bookings_not_canceled','reservation_status'
  - Plot the heatmap and see the correlation.
                                    
- Model Building

  - Two different Train Test Split (2:1 and 4:1)
  - Two different application data mining (Orange and Jupyter notebook)
  - Using pipeline for model building
      - scaling for numerical features
      - label encoder for categorical features
   - Creating base model with two algorithm (Logistic Regression, Decision Tree Classifier)
   - Checking evaluation matrix
   - Hyperparameter tuning on best performing model
   - Checking evaluation matrix on the tuned model
   - Export the model with the best accuracy score
  
  - Data Product Building Using Streamlit
  
    - Data Description
    - Data Visualization
    - Data Prediction
    
## Experiment setting of the data mining 

### 1. Descriptive data mining

### 2. Predictive data mining

#### Orange 
![image](https://user-images.githubusercontent.com/92434335/146476267-7bd591b4-5e0e-4e56-b17e-5239d2ee3913.png)


## Results and analysis of the performance comparison

## Data product

Data product using Streamlit: [here](https://share.streamlit.io/jinglongchia/sqit5033hotelbooking/main/app.py)

https://user-images.githubusercontent.com/92434335/146469968-740dd315-3026-410f-ba95-0b46bd3de3e2.mp4

Some description on data about where the data from, what are the contains in the data.

https://user-images.githubusercontent.com/92434335/146469985-102d24f5-c536-48f9-ab66-9e29fcbc1d7a.mp4

Analyze the hotel booking demand data accross various Exploratory data analysis using bar chart.

https://user-images.githubusercontent.com/92434335/146470004-7517a26d-e441-4b2d-a626-8490339634f2.mp4

Predict the possible canceled hotel booking using Decision Tree classfication algorithm.

## Conclusion and reflection

