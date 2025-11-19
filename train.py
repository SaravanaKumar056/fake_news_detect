import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string
import pickle

print("Loading datasets...")
df_fake = pd.read_csv("dataset/Fake.csv")
df_true = pd.read_csv("dataset/True.csv")

df_fake["class"] = 0
df_true["class"] = 1

print("Merging and shuffling data...")
df_merge = pd.concat([df_fake, df_true], axis=0)
df = df_merge.drop(["title", "subject", "date"], axis=1)

df = df.sample(frac=1).reset_index(drop=True)

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

print("Cleaning text data (this might take a moment)...")
df["text"] = df["text"].apply(wordopt)

x = df["text"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print("Vectorizing text...")
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train Logistic Regression (I use Logistic Regression beacuse it is the best performing model)
print("Training Logistic Regression model...")
LR = LogisticRegression()
LR.fit(xv_train, y_train)

# Save the Model and Vectorizer
print("Saving model.pkl and vectorizer.pkl...")
with open('model.pkl', 'wb') as f:
    pickle.dump(LR, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorization, f)

print("Success! You can now run app.py.")