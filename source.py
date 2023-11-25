#!/usr/bin/env python
# coding: utf-8

# # {Project Title}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# Car Crash Rates - Car crash rates are increasingly on the rise so what are the major reasons for the number of crashes?

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# What are the major factors that are causing an increase in car crashes?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# My hypothesis is that texting and substance abuse are the leading causes of increased crashes.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# https://www.kaggle.com/datasets/joebeachcapital/car-crashes
# 
# https://crashviewer.nhtsa.dot.gov/CrashAPI  (There are several API calls that I will be utilizing here)
# 
# http://data.ctdata.org/dataset/motor-vehicle-accidents
# 
# I will be relating these data sets to identify commonalities amongst them to ensure accurate results.
# 
# 
# 
# 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# This demonstrates the call, but I will be using the data to identify common trends as to what causes the most accidents and crashes.

# In[1]:


# Start your code here

import requests

url = "https://crashviewer.nhtsa.dot.gov/CrashAPI/crashes/GetCaseList?states=1,51&fromYear=2014&toYear=2015&minNumOfVehicles=1&maxNumOfVehicles=6&format=json"

response = requests.get(url)

if response.status_code == 200:
    # The request was successful, and the response is in JSON format
    data = response.json()
    # Now you can work with the JSON data
    print(data)
else:
    print(f"Request failed with status code: {response.status_code}")


# ## Checking to See if any Duplicate or Missing Values

# In[2]:


# Checking for Duplicate and Missing Values for the cleanmotorvehicleaccidents.csv file
import pandas as pd

CMVAFilePath = 'cleanmotorvehicleaccidents.csv'
CMVAData = pd.read_csv(CMVAFilePath)

# Check for duplicates
CMVADuplicates = CMVAData[CMVAData.duplicated()]
if not CMVADuplicates.empty:
    print("Duplicate rows found:")
    print(CMVADuplicates)
else:
    print("No duplicate rows found.")

# Check for missing values
CMVAMissingValues = CMVAData.isnull().sum()
if CMVAMissingValues.sum() > 0:
    print("Columns with missing values:")
    print(CMVAMissingValues[CMVAMissingValues > 0])
else:
    print("No missing values found.")


# In[3]:


# Checking for Duplicate and Missing Values for the Motor_Vehicle_Collisions_-_Crashes.csv file
import pandas as pd

MVCCFilePath = 'Motor_Vehicle_Collisions_-_Crashes.csv'
MVCCData = pd.read_csv(MVCCFilePath)

# Check for duplicates
MVCCDuplicates = MVCCData[MVCCData.duplicated()]
if not MVCCDuplicates.empty:
    print("Duplicate rows found:")
    print(MVCCDuplicates)
else:
    print("No duplicate rows found.")

# Check for missing values
MVCCMissingValues = MVCCData.isnull().sum()
if MVCCMissingValues.sum() > 0:
    print("Columns with missing values:")
    print(MVCCMissingValues[MVCCMissingValues > 0])
else:
    print("No missing values found.")


# ### Since there are a multitude of missing values, I decided to keep them since I felt they were not construing the actual data. While extra data may not be present, it doesn't mean that the accidents didn't happen.

# In[4]:


import requests
import pandas as pd
from pandas.core.common import flatten

url = "https://crashviewer.nhtsa.dot.gov/CrashAPI/crashes/GetCaseList?states=1,51&fromYear=2014&toYear=2015&minNumOfVehicles=1&maxNumOfVehicles=6&format=json"

response = requests.get(url)

if response.status_code == 200:
    # The request was successful, and the response is in JSON format
    data = response.json()
    
    # Flatten the JSON data to handle potential nested structures
    flattenedData = list(flatten(data))

    # Convert flattened JSON data to a pandas DataFrame
    CVData = pd.DataFrame([flattenedData])
    
    # Check for duplicates
    CVDuplicates = CVData[CVData.duplicated()]
    if not CVDuplicates.empty:
        print("Duplicate rows found:")
        print(CVDuplicates)
    else:
        print("No duplicate rows found.")

    # Check for missing values
    CVMissingValues = CVData.isnull().sum()
    if CVMissingValues.sum() > 0:
        print("Columns with missing values:")
        print(CVMissingValues[CVMissingValues > 0])
    else:
        print("No missing values found.")
        
else:
    print(f"Request failed with status code: {response.status_code}")


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

filePath = 'cleanmotorvehicleaccidents.csv'
data = pd.read_csv(filePath)

# Count occurrences of each 'Abused Substance'
substance_counts = data['Abused Substance'].value_counts()

plt.figure(figsize=(10, 6))
substance_counts.plot(kind='bar')
plt.title('Number of Accidents for Each Abused Substance')
plt.xlabel('Abused Substance')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #### The graph above correlates the number of accidents that occurred when the driver was intoxicated via alcohol vs drugs. I used a bar chart to easily show the numbers.

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

filePath = 'Motor_Vehicle_Collisions_-_Crashes.csv'
data = pd.read_csv(filePath)

# Group by 'CONTRIBUTING FACTOR VEHICLE 1' and count occurrences of fatalities
fatalities_per_factor = data['CONTRIBUTING FACTOR VEHICLE 1'].loc[data['NUMBER OF MOTORIST KILLED'] > 0].value_counts()

plt.figure(figsize=(12, 6))
fatalities_per_factor.plot(kind='bar', color='blue')
plt.title('Number of Motorists Killed by Contributing Factor')
plt.xlabel('Contributing Factor Vehicle 1')
plt.ylabel('Number of Motorists Killed')
plt.xticks(rotation=90)
plt.yticks(range(0, max(fatalities_per_factor)+1, 50))
plt.tight_layout()
plt.show()


# #### This chart goes even further beyond and breaks down crashes into more specific categories. I chose a histogram because it suits the type of data I wish to display

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

filePath = 'Motor_Vehicle_Collisions_-_Crashes.csv'
data = pd.read_csv(filePath)

killed_motorists_data = data[data['NUMBER OF MOTORIST KILLED'] > 0]

killed_per_borough = killed_motorists_data['BOROUGH'].value_counts()

plt.figure(figsize=(10, 6))
killed_per_borough.plot(kind='bar', color='purple')
plt.title('Number of Motorists Killed by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Motorists Killed')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Again, I used a Bar Graph to easily show the number of deaths per Borough. Surprisingly, the most deaths occurred in Brooklyn and Queens instead of Manhattan, which is what I originally guessed would be the highest.

# In[8]:


import requests
import matplotlib.pyplot as plt

url = "https://crashviewer.nhtsa.dot.gov/CrashAPI/crashes/GetCaseList"

# Parameters for the API call
params = {
    'states': '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51',
    'fromYear': '2014',
    'toYear': '2015',
    'minNumOfVehicles': '1',
    'maxNumOfVehicles': '6',
    'format': 'json'
}

try:
    response = requests.get(url, params=params)

    if response.status_code == 200:
   
        data = response.json()

        # Get 'fatals' and 'persons' columns data
        fatals = []
        persons = []

        # Get 'fatals' and 'persons' from nested lists of dictionaries
        for sublist in data['Results']:
            for entry in sublist:
                if 'Fatals' in entry and 'Persons' in entry:
                    fatals.append(entry['Fatals'])
                    persons.append(entry['Persons'])

        # Time to plot
        plt.figure(figsize=(8, 6))
        plt.scatter(fatals, persons, color='blue', alpha=0.5)
        plt.title('Fatal Crashes By Car Occupants')
        plt.xlabel('Fatalaties')
        plt.ylabel('People in Car')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Request failed with status code: {response.status_code}")

except requests.RequestException as e:
    print(f"Request failed: {e}")


# #### This chart shows that the more fatalaties occur when there are more people in the vehicle. I had a lot of issues with the API call (hence the try/except) but I was able to narrow down the call with the right parameters being passed.

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the data from the CSV file - only first 5000 rows.
file_path = 'Motor_Vehicle_Collisions_-_Crashes.csv'
columns_to_use = ['CRASH DATE', 'CRASH TIME', 'BOROUGH', 'LATITUDE', 'LONGITUDE', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED', 'CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2', 'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5', 'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5']
data = pd.read_csv(file_path, usecols=columns_to_use, nrows=5000)

features = data.drop('CRASH DATE', axis=1)  # Remove 'CRASH DATE' due to type issues
target = data['NUMBER OF PERSONS INJURED'] 

# 80% training, 20% test
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)


numeric_features = ['LATITUDE', 'LONGITUDE', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']
numeric_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['CRASH TIME', 'BOROUGH', 'CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2', 'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5', 'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5']
categorical_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transform, numeric_features),
        ('cat', categorical_transform, categorical_features)
    ])

# Create a complete pipeline 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier() 

# full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# training data
pipeline.fit(train_features, train_target)

# test data
accuracy = pipeline.score(test_features, test_target)
print(f"Accuracy on test set: {accuracy}")


# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


file_path = 'cleanmotorvehicleaccidents.csv'
columns_to_use = ['Town', 'FIPS', 'Year', 'Abused Substance', 'Outcome', 'Measure Type', 'Variable', 'Value']
data = pd.read_csv(file_path, usecols=columns_to_use)

# Define features and target variable
features = data.drop('Outcome', axis=1)  # Assuming 'Outcome' is the target variable
target = data['Outcome']

# Split the data into training and test sets
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)

# Preprocessing pipelines for numerical and categorical features
numeric_features = ['Year', 'Value']  
numeric_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['Town', 'FIPS', 'Abused Substance', 'Measure Type', 'Variable']  
categorical_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transform, numeric_features),
        ('cat', categorical_transform, categorical_features)
    ])

# Define the model
model = RandomForestClassifier()

# Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Fit the pipeline on the training data
pipeline.fit(train_features, train_target)

# Evaluate the pipeline on the test data
predictions = pipeline.predict(test_features)
accuracy = accuracy_score(test_target, predictions)
print(f"Accuracy on test set: {accuracy}")
print("Training Features:")
print(train_features.head())  # Display the first few rows of the training features
print("\nTraining Target:")
print(train_target.head())  # Display the first few rows of the training target

print("\nTest Features:")
print(test_features.head())  # Display the first few rows of the test features
print("\nTest Target:")
print(test_target.head())  # Display the first few rows of the test target


# In[12]:


import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

url = "https://crashviewer.nhtsa.dot.gov/CrashAPI/crashes/GetCaseList"

# Parameters for the API call
params = {
    'states': '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51',
    'fromYear': '2014',
    'toYear': '2015',
    'minNumOfVehicles': '1',
    'maxNumOfVehicles': '6',
    'format': 'json'
}

try:
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        fatals = []
        persons = []

        # Get 'fatals' and 'persons' from nested lists of dictionaries
        for sublist in data['Results']:
            for entry in sublist:
                if 'Fatals' in entry and 'Persons' in entry:
                    fatals.append(entry['Fatals'])
                    persons.append(entry['Persons'])

        # Split the data into training and test sets
        fatals_train, fatals_test, persons_train, persons_test = train_test_split(fatals, persons, test_size=0.2, random_state=42)

        # Plotting for both training and test data
        plt.figure(figsize=(8, 6))
        plt.scatter(fatals_train, persons_train, color='blue', alpha=0.5, label='Training Data')
        plt.scatter(fatals_test, persons_test, color='red', alpha=0.5, label='Test Data')
        plt.title('Fatal Crashes By Car Occupants')
        plt.xlabel('Fatalities')
        plt.ylabel('People in Car')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Request failed with status code: {response.status_code}")

except requests.RequestException as e:
    print(f"Request failed: {e}")


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# Stack Overflow, Python Documentation, the API information from the site at the top, Geeks for Geeks, various forums, W3Schools, Sklearn model documentation

# In[13]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

