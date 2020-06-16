# Rémi Wong

# The cancer prediction dataset was hosted on my google drive and the codes were run on google colab
from google.colab import drive
drive.mount('/content/drive')

!pip install keras

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

dataframeURL = '/content/drive/My Drive/Tensorflow/risk_factors_cervical_cancer_cleaned2.csv'


dataframe = pd.read_csv(dataframeURL, encoding='utf-8')

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

print(train)

train.shape

train.head(10)



# Transform Pandas Dataframe to tf.data dataset
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('DxCancer')
  excessFeatures = dataframe.pop('DxCIN')
  excessFeatures = dataframe.pop('DxHPV')
  excessFeatures = dataframe.pop('Dx')

  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


batch_size = 8 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#View the breakdown of dataset
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['Age'])
  print('A batch of Cancer:', label_batch )

#set all values to float64
tf.keras.backend.set_floatx('float64')


feature_columns = []

#push numeric cols to the feature layer
for header in ['Age', 'sexualPartners', 'Firstsexualintercourse', 'Numofpregnancies', 'Smokes', 'Smokes2', 'HormonalContraceptives', 'IUD', 'STDscondylomatosis', 'STDscervicalcondylomatosis', 'STDsvaginalcondylomatosis', 'STDsvulvo-perinealcondylomatosis', 'STDssyphilis', 'STDspelvicinflammatorydisease', 'STDsgenitalherpes', 'STDsmolluscumcontagiosum', 'STDsAIDS', 'STDsHIV', 'STDsHepatitisB', 'STDsHPV', 'STDsNumberofdiagnosis']:
  feature_columns.append(feature_column.numeric_column(header))


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#batch_size = 548
train_ds = df_to_dataset(train, batch_size=548)
val_ds = df_to_dataset(val, shuffle=False, batch_size=138)
test_ds = df_to_dataset(test, shuffle=False, batch_size=172)


#Build the Neural Network
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])

#compile and optimze model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

#validate model and repeat training with 10 epochs
model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)