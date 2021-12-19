# A211 SQIT5033 Hotel Booking Demand Project

![alt text](https://www.hotelspeak.com/wp-content/uploads/2019/05/hotel-direct-booking-strategy.jpg)

## Introduction to Dataset

We will use the Hotel Booking Demand dataset from the Kaggle.

You can download it from here: https://www.kaggle.com/jessemostipak/hotel-booking-demand

This data set contains booking information for a city hotel and a resort hotel and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things. All personally identifying information has from the data.


## Problem Statement

With the increase trend of cancellation from year to year, some hotel have think that high cancellation in hotel is the new norm of the industry which is a completely wrong approach, one out of four hotel guests are cancelling hotel booking ahead of a stay. This cancellation trend has effect the hotel not being able to accurately forecast occupancy within their revenue management, and the trend of cancellation also have causes hotel loss in opportunity cost (unsold room due to cancellation).


## Hypothesis

1. Cancellation booking has nothing to do with uncontrollable situations such as floods and epidemics.

2. The hotel has no promotion or reputation damage that affects hotel check-in and cancellation booking.


## Goals

1. The Goals of this project is to find out the characteristic of customers who cancelled and finding a pattern in cancelled booking by doing an exploratory data analysis.

2. Building descriptive data mining to determine the data regularities and predictive data mining to predict cancellation.

3. Build and Deploy web application / dashboard using Streamlit from predictive data mining, that can predict of cancellation based on user input.


## Exploratory data analysis

Exploratory data analysis can be review at: [here](https://github.com/JingLongChia/A211-SQIT5033-Hotel-Booking-Demand-Project/blob/main/Exploratory%20Data%20Analysis/Exploratory%20Data%20Analysis.ipynb)

### Booking Hotel & Cancellation

![image](https://user-images.githubusercontent.com/92434335/146670213-07b8b340-4c39-41e4-83f4-dcc2f426c787.png)

Based on the graph above, shown the hotel booking cancellation for both type of hotel is less than 50%, this means the customer will not cancel booking casually.

City Hotel Booking have 42% cancellation Rate while Resort Hotel Booking have 28% cancellation Rate.

This is probably because most of the city hotel bookings are for work needs. Sometimes the booking may be cancelled if the time is too busy. Most of the customers of Resort hotel come for vacation, I believe that few people take the initiative to cancel a pleasant holiday trip.


### Arrival Month & Cancellation

![image](https://user-images.githubusercontent.com/92434335/146670335-0cdfad9f-a405-48e9-9b1a-c78547fd1767.png)

Look at the table above, majority of the month has a cancellation rate around 30 to 40 percent.

The difference is not big, the small changes are likely to be due to the seasons and holidays.


### Deposit Type & Cancellation

![image](https://user-images.githubusercontent.com/92434335/146670467-5a2df63c-718e-4fb3-957f-1d8369331aea.png)

This Dataset has 3 kinds of deposit type NO Deposit, NO Refund, and Refundable, all of the name is kind of self explanatory, based on our analysis we found out that:

- No Refund Booking has the highest cancellation rate at 99.4%

- No Deposit has cancellation rate of 28.3 %

- While Refundable has cancellation rate around 22%

For the hotels this is nothing alarming since they don't lose revenue when no refund booking is canceled, but it's always a good practice to question something is extraordinary, why does non refundable booking are most likely to be canceled? isn't just like wasting money cancelling your non refundable booking. To answer that question let's look at the median lead time of each deposit type.


### Market Segment & Cancellation

![image](https://user-images.githubusercontent.com/92434335/146670894-23c3f582-029f-4aed-b2e2-a8808de3c5dd.png)

- From our analysis we see that corporate , Direct, and Aviation has a cancellation rate around 18 - 22 % of their booking
- 
- Travel Agent (Online / Offline) has a cancellation rate around 34 - 36 %
- 
- Lastly Group has the highest cancellation rate around 61 %

Based on this we conclude that group booking are the market segment that's most likely to be canceled compared to other market segment while Direct has the lowest cancellation rate at 15% (Outside Complimentary).


### Repeated Guest & Cancellation¶

![image](https://user-images.githubusercontent.com/92434335/146670935-031f83a4-5112-45b5-a0b1-4bd883ece803.png)

In this dataset we only have around 3% of repeated guest, tho we still see the difference of cancellation pattern in both repeated guest and non repeated guest.

- Repeated Guest has cancellation rate around 14%

- Non Repeated Guest are more than 2X more likely to cancelled the booking compared to repeated guest

In conclusion, Repeated Guest are more likely to confirm their booking compared to non repeated guest.


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

### 3. Feature Selection

Feature selection is a very important part and a very difficult one. 

- Now let’s create some new features.

  - We have two features in our dataset reserved_room_type and another is assigned_room_type. We will make the new feature let’s call it Room which will contain 1 if the guest was assigned the same room that was reserved else 0. Guest can cancel the booking if he did not get the same room.

  - Another feature will be net_cancelled. It will contain 1 If the current customer has canceled more bookings in the past than the number of bookings he did not cancel, else 0.
  - 
```Python
# Make the new column which contain 1 if guest received the same room which was reserved otherwise 0
df_subset['Room'] = 0
df_subset.loc[ df_subset['reserved_room_type'] == df_subset['assigned_room_type'] , 'Room'] = 1


## Make the new column which contain 1 if the guest has cancelled more booking in the past
## than the number of booking he did not cancel, otherwise 0

df_subset['net_cancelled'] = 0
df_subset.loc[ df_subset['previous_cancellations'] > df_subset['previous_bookings_not_canceled'] , 'net_cancelled'] = 1
```

Now remove these unnecessary features.

```Python
## Remove the less important features
df_subset = df_subset.drop(['arrival_date_year','arrival_date_week_number','arrival_date_day_of_month',
                            'arrival_date_month','assigned_room_type','reserved_room_type','reservation_status_date',
                            'previous_cancellations','previous_bookings_not_canceled'],axis=1)
```

Let’s also remove the reservation_status. Even though it is a very important feature, but it already has information about canceled booking. Further, It can only have information after the booking was canceled or the guest checked in. So it will not be useful to use this feature in our predictive model. Because for the future prediction we won’t have information about the reservation status.

```Python
## Remove reservation_status column
## because it tells us if booking was cancelled 
df_subset = df_subset.drop(['reservation_status'], axis=1)
```
Let’s plot the heatmap and see the correlation

```Python
## Plot the heatmap to see correlation with columns
import dython
from dython.nominal import associations
associations(df_subset, figsize = (40, 20))
plt.show()
```
![A211-SQIT5033-Hotel-Booking-Demand-Project_Predictive Hotel (Train2Test1) ipynb at main · JingLongCh](https://user-images.githubusercontent.com/92434335/146694734-3b11849a-f279-4e25-aafc-8321e0de49b9.png)

We can see our new features, Room and net_cancelled have a higher correlation with is_cancelled than most of the other columns.

## Descriptive and predictive data mining solution

### 1. Descriptive data mining

### 2. Predictive data mining

  - Plot the heatmap and see the correlation and choose the data to predict.
                                    
- Model Building

  - Two different Train Test Split (2:1 and 4:1)
  - Two different application data mining (Orange and Jupyter notebook)
  - Using pipeline for model building
      - scaling for numerical features
      - label encoder for categorical features
   - Creating base model with two algorithm (Logistic Regression, Decision Tree Classifier)
   - Checking evaluation matrix
   - Hyperparameter tuning on every model
   - Checking evaluation matrix on the tuned model
   - Export the model with the best accuracy score
  
  - Data Product Building Using Streamlit
  
    - Data Description
    - Data Visualization
    - Data Prediction
    
## Experiment setting of the data mining 

### 1. Descriptive data mining

### 2. Predictive data mining



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

