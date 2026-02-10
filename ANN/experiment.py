import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
from sklearn.preprocessing import OneHotEncoder
data=pd.read_csv(r"C:\Users\Admin\Desktop\Learning\ML\ANN\Xl_File\Churn_Modelling.csv")
# print(data.head())

## drop method

data=data.drop(["RowNumber","CustomerId","Surname"],axis=1)
# print(data.head())

## label encounder is  used to change the o to 1 like women is o and men is 1 

gender_label=LabelEncoder()
data["Gender"]=gender_label.fit_transform(data["Gender"])
# print(data.head())

# one hot encoder

one_hot_encoder=OneHotEncoder()
geo_coder=one_hot_encoder.fit_transform(data[["Geography"]])
# print(geo_coder)

one_hot_geo=one_hot_encoder.get_feature_names_out(["Geography"])
# print(one_hot_geo)
geo_hot_df=pd.DataFrame(geo_coder.toarray(),columns=one_hot_geo)
# print(geo_hot_df)
data=pd.concat([data.drop(["Geography"],axis=1),geo_hot_df],axis=1)
# print(data.head())

# with open("label_encounder","wb") as file:
#     pickle.dump(gender_label,file)
# with open("onehot_encoder_geo.pkl","wb") as file:
#     pickle.dump(one_hot_encoder,file)

## DiVide the dataset into indepent and dependent features
X=data.drop('Exited',axis=1)
y=data['Exited']

## Split the data in training and tetsing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

## Scale these features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
# print(X_train)
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)

# print(data.head())


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime
(X_train.shape[1],)

model=Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)), ## HL1 Connected wwith input layer
    Dense(32,activation='relu'), ## HL2
    Dense(1,activation='sigmoid')  ## output layer
]

)

model.summary()

import tensorflow
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)
loss=tensorflow.keras.losses.BinaryCrossentropy()
loss

## compile the model
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])

## Set up the Tensorboard
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)


## Set up Early Stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

### Train the model
history=model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)

model.save('model.h5')

# ## Load Tensorboard Extension
# print(%load_ext tensorboard)

# %tensorboard --logdir logs/fit


# How to log call in the vs code first open the terminal and run the comment 
### tensorboard --logdir logs/fit