## classification

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Encode target labels

@st.cache_data
def load_data():
    iris=load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['species']= iris.target
    return df, iris.target_names

df, target_names= load_data()
le = LabelEncoder()
y = le.fit_transform(df["species"])
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(df.iloc[:,:-1], y)

st.sidebar.title("Input Feautures")

sepal_length= st.sidebar.slider("Sepal length",float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width= st.sidebar.slider("Sepal width",float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length= st.sidebar.slider("Petal length",float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width= st.sidebar.slider("Petal width",float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

# Prepare input
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
df_new = pd.DataFrame(input_data, columns=df.columns[:-1])

# Prediction
predict = model.predict(df_new)
predict_species = target_names[predict[0]]

# Probabilities
probs = model.predict_proba(df_new)
probs_df = pd.DataFrame(probs[0], index=target_names, columns=["Probability"])

# Display results in Streamlit
st.title("Prediction Results")
st.write(f"🌸 The Predicted Iris is **{predict_species}**")

st.subheader("Prediction Probabilities")
st.bar_chart(probs_df)
