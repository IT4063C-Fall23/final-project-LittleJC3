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

# In[3]:


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


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[10]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

