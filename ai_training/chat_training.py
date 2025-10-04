# write a script for training a neural network to classify activities based on imu sensor data
# dataset: ai_training/dataset/our_dataset.csv
# data structure: Timestamp,Activity_ID,Activity_Type,Ax,Ay,Az,Gx,Gy,Gz

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 
from tensorflow.keras.utils import to_categorical
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load dataset
data_path = "ai_training/dataset/out_dataset.csv"



if not os.path.exists(data_path):
    logger.error(f"Dataset not found at {data_path}")
    exit(1)



data = pd.read_csv(data_path)
logger.info(f"Dataset loaded with shape: {data.shape}")
# Preprocess data
# Handle missing values
data = data.dropna()
logger.info(f"Dataset shape after dropping NA: {data.shape}")
# Encode labels
label_encoder = LabelEncoder()
data['Activity_Type'] = label_encoder.fit_transform(data['Activity_Type'])
num_classes = len(label_encoder.classes_)
logger.info(f"Number of activity classes: {num_classes}")


# Normalize sensor data
sensor_columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
data[sensor_columns] = (data[sensor_columns] - data[sensor_columns].mean()) / data[sensor_columns].std()
logger.info("Sensor data normalized")   
# Apply low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a 
def lowpass_filter(data, cutoff=3.0, fs=50.0, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
for col in sensor_columns:
    data[col] = lowpass_filter(data[col])
logger.info("Low-pass filter applied to sensor data")
# Create sequences for LSTM
def create_sequences(df, seq_length=50):
    sequences = []
    labels = []
    for i in range(len(df) - seq_length):
        seq = df[sensor_columns].iloc[i:i+seq_length].values
        label = df['Activity_Type'].iloc[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)
seq_length = 50
X, y = create_sequences(data, seq_length)
logger.info(f"Created sequences with shape: {X.shape}, Labels shape: {y.shape}")
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)
logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(seq_length, len(sensor_columns)), return_sequences=True))
model.add(LSTM(32))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Train model
history = model.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_split=0.2)
# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
logger.info("Classification Report:")
logger.info(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Save model
model.save("ai_training/activity_classifier_model.h5")
logger.info("Model saved to ai_training/activity_classifier_model.h5")      
# Visualize some data
#same but for human movements datset
data = pd.read_csv("ai_training/dataset/all_activity_data.csv")
#how data are saved: Participant_ID,Activity_Type,Accel_X,Accel_Y,Accel_Z,Gyro_X,Gyro_Y,Gyro_Z,Timestamp
#first line: 1,Walking,-6.752,-1.462,6.634,50.458,-119.29,330.14,1727961275         
