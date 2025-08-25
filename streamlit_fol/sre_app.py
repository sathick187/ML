import streamlit as st
import pandas as pd
import numpy as np

## title

st.title("Welcome to my Web Page")


### Write

st.write("Hello guys")

# ### DATA frame
# df=pd.DataFrame({"name":["Sathick","dharshini"],"age":[25,23]})
# st.write(df)

# ### line chart

# line=pd.DataFrame(np.random.randn(20,3),columns=["a","b","c"])
# st.line_chart(line)

### text input
name=st.text_input("enter your name: ")
st.write (f"Hello ,{name}")

## slider
age=st.slider("choose your age: ",0,100,25)
st.write(f"your age is,{age}")

## choice

a=["Python","Java","C++"]
b=st.selectbox(f"selcet the course,",a)
st.write(b)

data={"name":["sk","sd"],"times":[2,5]}
df=pd.DataFrame(data)
st.write(df)
e=df.to_csv("new.csv")

### upload file

upload=st.file_uploader("choose your csv file",type="csv")
if upload:
    b=pd.read_csv(upload)
    st.write(b)
