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

### 1. Booking Hotel & Cancellation

![image](https://user-images.githubusercontent.com/92434335/146670213-07b8b340-4c39-41e4-83f4-dcc2f426c787.png)

Based on the graph above, shown the hotel booking cancellation for both type of hotel is less than 50%, this means the customer will not cancel booking casually.

City Hotel Booking have 42% cancellation Rate while Resort Hotel Booking have 28% cancellation Rate.

This is probably because most of the city hotel bookings are for work needs. Sometimes the booking may be cancelled if the time is too busy. Most of the customers of Resort hotel come for vacation, I believe that few people take the initiative to cancel a pleasant holiday trip.


### 2. Arrival Month & Cancellation

![image](https://user-images.githubusercontent.com/92434335/146670335-0cdfad9f-a405-48e9-9b1a-c78547fd1767.png)

Look at the table above, majority of the month has a cancellation rate around 30 to 40 percent.

The difference is not big, the small changes are likely to be due to the seasons and holidays.


### 3. Deposit Type & Cancellation

![image](https://user-images.githubusercontent.com/92434335/146670467-5a2df63c-718e-4fb3-957f-1d8369331aea.png)

This Dataset has 3 kinds of deposit type NO Deposit, NO Refund, and Refundable, all of the name is kind of self explanatory, based on our analysis we found out that:

- No Refund Booking has the highest cancellation rate at 99.4%

- No Deposit has cancellation rate of 28.3 %

- While Refundable has cancellation rate around 22%

For the hotels this is nothing alarming since they don't lose revenue when no refund booking is canceled.


### 4. Market Segment & Cancellation

![image](https://user-images.githubusercontent.com/92434335/146670894-23c3f582-029f-4aed-b2e2-a8808de3c5dd.png)

- From our analysis we see that corporate , Direct, and Aviation has a cancellation rate around 18 - 22 % of their booking

- Travel Agent (Online / Offline) has a cancellation rate around 34 - 36 %

- Lastly Group has the highest cancellation rate around 61 %

Based on this we conclude that group booking are the market segment that's most likely to be canceled compared to other market segment while Direct has the lowest cancellation rate at 15% (Outside Complimentary).


### 5. Repeated Guest & Cancellation

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

- Plot the heatmap and see the correlation and choose the data to descriptive.
                                    
- Model Building

  - Two different Train Test Split (2:1 and 4:1)
  - Using Jupyter notebook with python language to do descriptive data mining.
  - Using pipeline for model building
      - scaling for numerical features
      - label encoder for categorical features
   - Creating base model with K-Means clustering algorithm.
   - Checking evaluation matrix
   - Hyperparameter tuning on every model
   - Checking evaluation matrix on the tuned model
   - Export the model with the best accuracy score

### 2. Predictive data mining

- Plot the heatmap and see the correlation and choose the data to predict.
                                    
- Model Building

  - Two different Train Test Split (2:1 and 4:1)
  - Two different application data mining (Orange and Jupyter notebook)
  - Using pipeline for model building
      - scaling for numerical features
      - label encoder for categorical features
   - Creating base model with two classification algorithm (Logistic Regression, Decision Tree Classifier)
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

Descriptive data mining can be review at: [here](https://github.com/JingLongChia/A211-SQIT5033-Hotel-Booking-Demand-Project/tree/main/Descriptive%20data%20mining)

Target will set as is_canceled and other data select for this Descriptive data mining based on the correlation on the Heatmaps.

![image](https://user-images.githubusercontent.com/92434335/146695024-9b8cf151-7194-440a-88bc-fed5b828a95d.png)

```Python
X = df_1[['hotel','lead_time','market_segment', 'deposit_type','customer_type','Room','net_cancelled']]
y = df_1['is_canceled']
```

For this Descriptive data mining will using two different dataset split which is 2:1 and 4:1

```Python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.33, random_state = 42)
#And
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.25, random_state = 42)
```
The data selected in X which the data be train an test which classify as categorical and numerical.
The categorical selected will be converting from categorical variables to numerical by using LabelEncoder from Sklearn to encode in an ordinal fashion.

```Python
cat_columns = ['hotel','market_segment','deposit_type','customer_type']
num_columns = ['lead_time','Room','net_cancelled']
```

Below is the pipeline for the LabelEncoder and the K-Means clustering algorithm.

- K-Means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster.

```Python
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

pipe_KM = Pipeline([
    ("prep", prepocessor),
    ("algo", KMeans(n_clusters=2))
])
```

After running the base model result of K-Means, of both two of dataset split (2:1 and 4:1).
We will run for Hyperparameter Tuning model for K-Means by using GridSearchCV for both two of dataset split (2:1 and 4:1).
GridSearchCV is a model selection step and this should be done after Data Processing tasks. 
It is always good to compare the performances of Tuned and Untuned Models. 
This will cost us the time and expense but will surely give us the best results. 
The scikit-learn API is a great resource in case of any help. It’s always good to learn by doing.

Below is the parameter for GridSearchCV K-Means.

```Python
param_KM = {
    'algo__max_iter': [300,600,900],
    'algo__n_init': [10,20,30],
    'algo__algorithm': ['auto']
}

model_KM = GridSearchCV(estimator=pipe_KM, param_grid=param_KM, cv = 3, n_jobs = -1, verbose = 1, scoring='accuracy')
model_KM.fit(X_train, y_train)
```
After running all the base model and tunnel model in different dataset split(2:1 and 4:1), we will compare both and select the best performing model.

### 2. Predictive data mining

Predictive data mining can be view at: [here](https://github.com/JingLongChia/A211-SQIT5033-Hotel-Booking-Demand-Project/tree/main/Predrictive%20data%20mining)

Target will set as is_canceled and other data select for this Predictive data mining based on the correlation on the Heatmaps.

![image](https://user-images.githubusercontent.com/92434335/146695024-9b8cf151-7194-440a-88bc-fed5b828a95d.png)

```Python
X = df_1[['hotel','lead_time','market_segment', 'deposit_type','customer_type','Room','net_cancelled']]
y = df_1['is_canceled']
```

For this Predictive data mining will using two different dataset split which is 2:1 and 4:1

```Python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.33, random_state = 42)
#And
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.25, random_state = 42)
```
The data selected in X which the data be train an test which classify as categorical and numerical.
The categorical selected will be converting from categorical variables to numerical by using LabelEncoder from Sklearn to encode in an ordinal fashion.

```Python
cat_columns = ['hotel','market_segment','deposit_type','customer_type']
num_columns = ['lead_time','Room','net_cancelled']
```
Below is the pipeline for the LabelEncoder and the logistic regression classification algorithm and decision tree classification algorithm.

- Logistic regression is a classification technique borrowed by machine learning from the field of statistics. Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The intention behind using logistic regression is to find the best fitting model to describe the relationship between the dependent and the independent variable.

- Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.

```Python
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

pipe_logreg = Pipeline([
    ("prep", prepocessor),
    ("algo", LogisticRegression())
])

pipe_DT = Pipeline([
    ("prep", prepocessor),
    ("algo", DecisionTreeClassifier())
])
```

After running the base model result of logistic regression and decision tree, of both two of dataset split (2:1 and 4:1).
We will run for Hyperparameter Tuning model for logistic regression and decision tree by using GridSearchCV for both two of dataset split (2:1 and 4:1).
GridSearchCV is a model selection step and this should be done after Data Processing tasks. 
It is always good to compare the performances of Tuned and Untuned Models. 
This will cost us the time and expense but will surely give us the best results. 
The scikit-learn API is a great resource in case of any help. It’s always good to learn by doing.

Below is the parameter for GridSearchCV logistic regression.

```Python
param_logreg = {
    'algo__penalty':['l2', 'l1', 'elasticnet'],
    'algo__C':[1.0, 2.0, 3.0, 4.0,5.0],
    'algo__class_weight':[None, 'balanced']
}
model_logreg = GridSearchCV(estimator=pipe_logreg, param_grid=param_logreg, cv = 3, n_jobs = -1, verbose = 1, scoring='accuracy')
model_logreg.fit(X_train, y_train)
```
Below is the parameter for GridSearchCV decision tree.

```Python
param_DT = {
    'algo__min_samples_split': [2,1,3,4,6,8,10,],
    'algo__max_depth': [None,1,2,4,8,10,12,14,18, 20],
    'algo__min_samples_leaf':[1,2,4,5,8]
}

model_DT = GridSearchCV(estimator=pipe_DT, param_grid=param_DT, cv = 3, n_jobs = -1, verbose = 1, scoring='accuracy')
model_DT.fit(X_train, y_train)
```
After running all the base model and tunnel model in different dataset split(2:1 and 4:1), we will compare both and select the best performing model.

## Results and analysis of the performance comparison

### 1. Descriptive data mining

In this part, the application used is Jupyter Notebook with Python language.
The Orange software cannot calculate accuarcy with dataset split, so is not include in this analysis.

- Below is the Base model Evaluation Matrix result for dataset split (2:1 and 4:1)

#### Dataset split (2:1)

![image](https://user-images.githubusercontent.com/92434335/146695881-6633ac31-6562-494f-96a3-f6c190842be6.png)

#### Dataset split (4:1)

![image](https://user-images.githubusercontent.com/92434335/146695948-71d945fa-deb6-4d20-983e-d28476334fe3.png)

Based on the result above, we can see that the Dataset split (2:1) have better Accucary	0.644891 compare to Dataset split (4:1) Accucary 0.355971.

After comparing for base model, we will look to other tunnel model.

- Below is the tunnel model Evaluation Matrix result for dataset split (2:1 and 4:1)

#### Dataset split (2:1)

![image](https://user-images.githubusercontent.com/92434335/146696212-38900de1-31f6-4e9d-bb8a-67491049ce38.png)

#### Dataset split (4:1)

![image](https://user-images.githubusercontent.com/92434335/146696172-0d1c6eff-7f3b-4980-812f-100e87b4a209.png)

Based on the result above show the Hyperparameter Tuning using GridSearchCV.
We can see that the Accucary for Dataset split (2:1) have increase from 0.644891 to 0.645043.
Accucary for Dataset split (4:1) have increase from 0.355971 to 0.644163.

Although the data set split (4:1) increased the most, but the accuracy of the dataset split (2:1) is the best performing model for Descriptive data mining.

- So, the best performing model for will be Descriptive data mining used:
  - Jupyter Notebook with Python language
  - K-Menas classification algorithm
  - Dataset split (2:1)

### 2. Predictive data mining

In this part, the application used is Jupyter Notebook with Python language and Orange Software.

- Below is the Base model Evaluation Matrix result for dataset split (2:1 and 4:1)

#### Orange Software Dataset split (2:1)

![image](https://user-images.githubusercontent.com/92434335/146696847-04d31a5e-1e9c-4e5e-8c2f-8681b971567f.png)

#### Jupyter Notebook Dataset split (2:1)

![image](https://user-images.githubusercontent.com/92434335/146696742-a47d8a18-5009-4200-93c3-fe81914168a2.png)

#### Orange Software Dataset split (4:1)

![image](https://user-images.githubusercontent.com/92434335/146696859-63d48d1a-d6df-476f-84df-3261a8be31e0.png)

#### Jupyter Notebook Dataset split (4:1)

![image](https://user-images.githubusercontent.com/92434335/146696716-cfa6adc5-0db4-451c-8457-3ff887863486.png)

#### Table of comparison

![image](https://user-images.githubusercontent.com/92434335/146696600-bfa21ae1-962d-4e57-800a-144187f87df2.png)

Based on the result above, we can see that the Accucary	for Dataset split (4:1) using Jupyter Notebook with decision tree classification algorithm have better Accucary 0.782862 compare to other.

After comparing for base model, we will look to other tunnel model.

- Below is the tunnel model Evaluation Matrix result for dataset split (2:1 and 4:1)

#### Orange Software Dataset split (2:1)

![image](https://user-images.githubusercontent.com/92434335/146696871-50b6c474-efb8-4fdb-bd26-7423f70d9ebe.png)

#### Jupyter Notebook Dataset split (2:1)

![image](https://user-images.githubusercontent.com/92434335/146696920-0a5c850d-d216-4dfe-846b-3292274a5a5d.png)

#### Orange Software Dataset split (4:1)

![image](https://user-images.githubusercontent.com/92434335/146696896-55dafdf2-27fb-48ce-a051-2a5bdff3b880.png)

#### Jupyter Notebook Dataset split (4:1)

![image](https://user-images.githubusercontent.com/92434335/146696931-8c2a607d-e923-4605-9783-45da00daf041.png)

#### Table of comparison

![image](https://user-images.githubusercontent.com/92434335/146696940-c8c3b4f9-c874-422c-b53d-39dcf9f2941b.png)

Based on the result above show the Hyperparameter Tuning using GridSearchCV.
We can see that the Accucary for Dataset split (2:1) using Jupyter Notebook with decision tree classification algorithm have better Accucary 0.7828 compare to other.

- So, the best performing model for will be Predictive data mining used:
  - Jupyter Notebook with Python language
  - Decision tree classification algorithm
  - Dataset split (2:1)

## Data product

Data product using Streamlit: [here](https://share.streamlit.io/jinglongchia/sqit5033hotelbooking/main/app.py)

https://user-images.githubusercontent.com/92434335/146469968-740dd315-3026-410f-ba95-0b46bd3de3e2.mp4

Some description on data about where the data from, what are the contains in the data.

https://user-images.githubusercontent.com/92434335/146469985-102d24f5-c536-48f9-ab66-9e29fcbc1d7a.mp4

Analyze the hotel booking demand data accross various Exploratory data analysis using bar chart.

https://user-images.githubusercontent.com/92434335/146470004-7517a26d-e441-4b2d-a626-8490339634f2.mp4

Predict the possible canceled hotel booking using Decision Tree classfication algorithm.

## Conclusion and reflection

### Exploratory Data Analysis

#### 1. Booking Hotel & Cancellation

City Hotel Booking have 42% cancellation Rate while Resort Hotel Booking have 28% cancellation Rate.
- This is probably because most of the city hotel bookings are for work needs. Sometimes the booking may be cancelled if the time is too busy. Most of the customers of Resort hotel come for vacation, I believe that few people take the initiative to cancel a pleasant holiday trip.

#### 2. Arrival Month & Cancellation

-Majority of the month has a cancellation rate around 30 to 40 percent.

The difference is not big, the small changes are likely to be due to the seasons and holidays.

#### 3. Deposit Type & Cancellation

- No Refund Booking has the highest cancellation rate at 99.4%

- No Deposit has cancellation rate of 28.3 %

- While Refundable has cancellation rate around 22%

For the hotels this is nothing alarming since they don't lose revenue when no refund booking is canceled.

### 4. Market Segment & Cancellation

- From our analysis we see that corporate , Direct, and Aviation has a cancellation rate around 18 - 22 % of their booking

- Travel Agent (Online / Offline) has a cancellation rate around 34 - 36 %

- Lastly Group has the highest cancellation rate around 61 %

Based on this we conclude that group booking are the market segment that's most likely to be canceled compared to other market segment while Direct has the lowest cancellation rate at 15% (Outside Complimentary).

### 5. Repeated Guest & Cancellation

- Repeated Guest has cancellation rate around 14%

- Non Repeated Guest are more than 2X more likely to cancelled the booking compared to repeated guest

In conclusion, Repeated Guest are more likely to confirm their booking compared to non repeated guest.

### Descriptive data mining

Best performing model for will be Descriptive data mining used:
  - Jupyter Notebook with Python language
  - K-Menas classification algorithm
  - Dataset split (2:1)

### Predrictive data mining

Best performing model for will be Predictive data mining used:
  - Jupyter Notebook with Python language
  - Decision tree classification algorithm
  - Dataset split (2:1)

### Reflection / Recommendation

#### Increase Direct Booking Market Segment

- From this dataset we see that direct booking has the least cancellation rate 15% (outside complimentary) compared to other market segment, with only being 10% of total booking market segment having more booking from direct market segment will likely to reduce the number of cancellation.

##### Strategy to increase Direct Booking

1. Leverage the power of a well optimized website

 - Visually attractive website
 - Offer & Ensure Best Rate Guarantee
 - Multilanguage & multi currency features
  
2. Increase Hotel Online Reputation

 - Almost 98% of travelers read hotel reviews and 80% of them consider them extremely important before making the final reservation. A one-point increase in a hotel’s average user rating on a 5-point scale (eg, 3.8 to 4.8) makes potential customers 13.5% more likely to book that hotel

3. Offer Loyalty program with difference

 - Incentivizing your guest with loyalty programs to book directly at the hotel website, by giving them points that could easily be redeemed not only at the hotel but at also at certain POS outlets
