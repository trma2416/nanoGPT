import os
import pickle
import requests
import numpy as np
import pandas as pd
import re
#we need to clean our input data
def clean_txt(text):
    text = str(text).strip() #remove leading and trailing whitespaces
    text = re.sub(r'\s+', '', text) #fix any unneccessary whitespace
    text = re.sub(r'^\x00-\x7F]+', '', text) #remove any characters non-ASCII
    return text

#read in our data
df = pd.read_csv("sample-data.csv")
#remove any null spaces
df.dropna(subset=["description"], inplace=True)

for col in ["id", 'description']:
    df[col] = df[col].apply(clean_txt)



