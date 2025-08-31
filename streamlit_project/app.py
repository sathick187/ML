## classification

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Encode target labels

@st.cache_data    # decorders    
def load_data():
    iris=load_iris()   ### inbulid functions 
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['species']= iris.target    ### target numeric values they are the 3 values so its taken of [0,1,2]
    return df, iris.target_names   ### target_names they are the 3 names in the flower like ['setosa' 'versicolor' 'virginica']

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



### the above function explanation  
# Your function is:

# @st.cache_data
# def load_data():
#     iris = load_iris()
#     df = pd.DataFrame(iris.data, columns=iris.feature_names)
#     df['species'] = iris.target
#     return df, iris.target_names

# 🔹 Step 1: What is load_iris()?

# load_iris() is a helper function from scikit-learn that loads a small sample dataset.
# It gives you information about 150 iris flowers (a type of flower).

# Each flower is measured by 4 numbers:

# Sepal length (cm)

# Sepal width (cm)

# Petal length (cm)

# Petal width (cm)

# And each flower belongs to one of 3 species:

# Setosa (0)

# Versicolor (1)

# Virginica (2)

# 🔹 Step 2: What’s inside iris?

# When you run:

# iris = load_iris()


# You get a Bunch object (like a dictionary).
# It has these important parts:

# iris.data → all flower measurements

# Shape: (150, 4)

# Example (first 5 rows):

# [[5.1 3.5 1.4 0.2]
#  [4.9 3.0 1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.0 3.6 1.4 0.2]]


# (each row = one flower, each column = one measurement)

# iris.feature_names → names of the 4 measurements

# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


# iris.target → species labels (numeric: 0,1,2)

# Shape: (150,)

# Example (first 10):

# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# (the first 50 flowers are Setosa = 0)

# iris.target_names → mapping from numbers to species names

# ['setosa' 'versicolor' 'virginica']


# iris.DESCR → text description of the dataset.

# iris.filename → path to dataset file.

# 🔹 Step 3: Convert into DataFrame
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['species'] = iris.target


# This creates a pandas DataFrame with 150 rows and 5 columns:

# sepal length (cm)	sepal width (cm)	petal length (cm)	petal width (cm)	species
# 5.1	3.5	1.4	0.2	0
# 4.9	3.0	1.4	0.2	0
# 4.7	3.2	1.3	0.2	0
# 4.6	3.1	1.5	0.2	0
# 5.0	3.6	1.4	0.2	0

# The first 4 columns are measurements.

# The last column species is the numeric label (0,1,2).

# 🔹 Step 4: Return values
# return df, iris.target_names


# df → the full dataset (150 flowers × 5 columns).

# iris.target_names → the mapping of labels:

# 0 = setosa

# 1 = versicolor

# 2 = virginica

# 🔹 Summary in Plain English

# load_iris() gives you a dataset of 150 flowers.

# Each flower has 4 measurements and belongs to 1 of 3 species.

# Your function makes this into a table (DataFrame) and also gives the species names.

# 👉 Think of it like a spreadsheet of flowers:

# Each row = one flower.

# Each column = one characteristic (length/width of sepal/petal, plus species).

# Would you like me to also explain why this dataset is so famous in Machine Learning (why everyone uses it for demos)?