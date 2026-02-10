📊 Customer Churn Prediction using ANN (TensorFlow & Keras)

This project builds an Artificial Neural Network (ANN) to predict customer churn (whether a customer will leave the bank or not) using TensorFlow/Keras.
It covers data preprocessing, feature engineering, model training, evaluation, and TensorBoard visualization.

📁 Project Overview

Customer churn prediction is a binary classification problem where:

1 → Customer exited

0 → Customer stayed

The model learns patterns from customer data such as age, balance, geography, gender, and credit score.

🛠️ Technologies Used

Python 3.x

Pandas

NumPy

Scikit-learn

TensorFlow / Keras

TensorBoard

VS Code

📂 Dataset

File: Churn_Modelling.csv

Columns used:

CreditScore

Geography

Gender

Age

Tenure

Balance

NumOfProducts

HasCrCard

IsActiveMember

EstimatedSalary

Exited (Target)

Dropped Columns:

RowNumber

CustomerId

Surname

These columns do not contribute to prediction.

🔄 Data Preprocessing Steps
1️⃣ Drop Unnecessary Columns
data.drop(["RowNumber","CustomerId","Surname"], axis=1)

2️⃣ Label Encoding (Gender)

Female → 0

Male → 1

Used because gender has only two categories.

3️⃣ One-Hot Encoding (Geography)

Converts categorical geography values into binary columns to avoid ordinal bias.

Example:

France → Geography_France = 1
Spain  → Geography_Spain = 1
Germany → Geography_Germany = 1

4️⃣ Feature Scaling
StandardScaler()


Scaling is mandatory for neural networks to ensure stable and fast learning.

5️⃣ Train-Test Split

80% Training

20% Testing

train_test_split(test_size=0.2, random_state=42)

🧠 ANN Model Architecture
Input Layer  → 64 neurons (ReLU)
Hidden Layer → 32 neurons (ReLU)
Output Layer → 1 neuron (Sigmoid)

Why this architecture?

ReLU avoids vanishing gradient

Sigmoid outputs probability (0–1)

Binary Crossentropy suits binary classification

⚙️ Model Compilation

Optimizer: Adam

Learning Rate: 0.01

Loss: Binary Crossentropy

Metric: Accuracy

⏹️ Early Stopping

Stops training when validation loss does not improve for 10 epochs.

Benefits:

Prevents overfitting

Saves best weights automatically

📊 TensorBoard Visualization (VS Code)
❌ Not Supported in .py files

Jupyter magic commands like:

%load_ext tensorboard
%tensorboard


❌ Do NOT work in VS Code .py files

✅ Correct Way (VS Code Terminal)
Step 1: Open Terminal
Ctrl + `

Step 2: Run TensorBoard
tensorboard --logdir logs/fit

Step 3: Open Browser
http://localhost:6006


You can now view:

Training vs Validation Loss

Training vs Validation Accuracy

Weight histograms

💾 Model Saving
model.save("model.h5")


Saved for:

Reuse

Deployment

API integration

💾 Scaler Saving
pickle.dump(scaler, open("scaler.pkl", "wb"))


Ensures same scaling during prediction.

📦 Installation (pip)

Run the following commands:

pip install pandas
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install tensorboard

🚀 How to Run the Project

Clone / Download the project

Place Churn_Modelling.csv correctly

Run the training script:

python churn_ann.py


Launch TensorBoard:

tensorboard --logdir logs/fit

🎯 Output

Trained ANN model (model.h5)

Scaler file (scaler.pkl)

TensorBoard logs (logs/fit/)

Accuracy & loss visualization

🧪 Future Improvements

Hyperparameter tuning

Dropout layers

Class imbalance handling

Flask / Streamlit deployment

Real-time prediction API

🧠 Interview Explanation (One Line)

This project predicts customer churn using an Artificial Neural Network with proper preprocessing, feature scaling, early stopping, and TensorBoard visualization to ensure optimal and stable model performance.

If you want, next I can:

✅ Add prediction script

✅ Add Streamlit UI

✅ Convert this into GitHub-ready structure

✅ Add diagrams for ANN

Just tell me 👍🔥