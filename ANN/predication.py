import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

model=load_model(r"C:\Users\Admin\Desktop\Learning\model.h5")
with open(r"C:\Users\Admin\Desktop\Learning\label_encounder","rb") as file:
    label=pickle.load(file)

with open(r"C:\Users\Admin\Desktop\Learning\onehot_encoder_geo.pkl","rb") as one_file:
    one_lable=pickle.load(one_file)

with open(r"C:\Users\Admin\Desktop\Learning\scaler.pkl","rb") as one_scalar:
    one_scale=pickle.load(one_scalar)

input_data={
    "CreditScore":600,
    "Geography":"France",
    "Gender":"Male",
    "Age":40,
    "Tenure":3,
    "Balance":60000,
    "NumOfProducts":2,
    "HasCrCard":1,
    "IsActiveMember":1,
    "EstimatedSalary":50000

}

geo_encoded=one_lable.transform([[input_data["Geography"]]]).toarray()
geo_df=pd.DataFrame(geo_encoded,columns=one_lable.get_feature_names_out(["Geography"]))
# print(geo_df)
input_df=pd.DataFrame([input_data])
input_df["Gender"]=label.transform(input_df["Gender"])

input_df=pd.concat([input_df.drop("Geography",axis=1),geo_df],axis=1)


##scaling input data

input_scaled=one_scale.transform(input_df)
# print(input_scaled)

#predicatiom

predication=model.predict(input_scaled)
# print(predication)    # op[[0.02972494]]

predication_pro=predication[0][0]

if predication_pro>0.5:
    print("yes  he is inside")
else:
    print("NO")