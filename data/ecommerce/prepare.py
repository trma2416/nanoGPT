import os
import openai
import time
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import re
import tiktoken


#we need to clean our input data
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
#dotenv_path = "Users/pc/Documents/CSPB/CSPB 3112/nanoGPT/.env"
load_dotenv(dotenv_path)
data_path = os.path.join(os.path.dirname(__file__), 'sample-data.csv')
if data_path:
    print("Hooray your data path works")
else: 
    raise NameError("no such file")

def clean_txt(text):
    text = str(text).strip() #remove leading and trailing whitespaces
    text = re.sub(r'\s+', ' ', text) #fix any unneccessary whitespace
    text = re.sub(r'^\x00-\x7F]+', '', text) #remove any characters non-ASCII
    return text
#used to split our description on each row into the new column titled items
def split(text):
    txt_lst = text.split("-")
    return txt_lst[0]

def split2(txt):
    txt_lst = txt.split("-")
    return txt_lst[1]
    
#format the data in a conversational format
def format_row(row):
    return f"Customer: Tell me about the {row['item']}.\nAgent: The {row['item']} is {row['description']}.\n"

#some of our descriptions lack correct sentence structure which is an issue for training
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def fix_grammar(txt: str) -> str:
    #edge case
    if not txt.strip():
        print("No text provided")
        return txt
    try:
        time.sleep(1)
        response = client.chat.completions.create(model="gpt-4-turbo", 
                                            messages=[{"role": "system", "content": ("You are a language expert, specializing in spelling correction and correcting sentence structure. "
                                                                                     "Return in this format: \n\ncustomer: <cleaned>\nagent: <cleaned>\n\n"
                                                                                     "Preserve as much of the vocabulary and technical terms as possible whilest maintaining articulate and clear phrases"
                                                                                     )},
                                                      {"role": "user", "content": txt}], 
                                            temperature=0.2,
                                            max_tokens=4000)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return txt


#read in our data
df = pd.read_csv(data_path)

#our item is hidden in our description so we need to extract that and create a new column
df["item"] = df["description"].apply(split)
df["description"] = df["description"].apply(split2)
df.dropna(subset=["description", "item"], inplace=True)
#clean up our data
for col in ["id", "description", "item"]:
    df[col] = df[col].apply(clean_txt)
    

#finally reformat it so it can be written into a text file
formatted = df.apply(format_row, axis=1) #apply to row via axis=1

with open("train.txt", "w", encoding = 'utf-8') as f:
    f.writelines(formatted)
print("Data has been reformatted!")

with open("train.txt", "r", encoding='utf-8') as file:
    original = file.read()
#if len(original) > 0:
    #print(original[:100]) #first 100 characters to ensure that original is stroring correct data
    #print("File is ready to make API calls!")
print("Length of original file: ", len(original))
cleaned_data = fix_grammar(original)
print("Data is cleaned and ready to be written into file!")
print("Length of cleaned data is: ", len(cleaned_data))

time.sleep(5)
with open("train.txt", "w", encoding='utf-8') as tf:
    tf.write(cleaned_data)
    
print("Your txt file is ready!")

## now we need to tokenize and split our data

inputfilepath= os.path.join(os.path.dirname(__file__), 'train.txt')

#split data into validation and trianing
with open(inputfilepath, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")


train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))