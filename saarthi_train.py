import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c','--config',type=str, default='external.json')
args = parser.parse_args()

config_file = args.config
def config_read(name):
  filename = name
  contents = open(filename).read()
  config = eval(contents)
  b = config['BATCH_SIZE']
  m = config['MAX_LEN']
  e = config['EPOCHS']
  f = config['FOLDER_PATH']
  t = config['TEST_PATH']
  return b,m,e,f,t

BATCH_SIZE, MAX_LEN, EPOCHS, FOLDER_PATH, TEST_PATH = config_read(config_file)

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
import transformers
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from transformers import TFRobertaModel
from keras.callbacks import CSVLogger
import json
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.metrics import f1_score
import warnings
import logging, sys
import os
from datetime import datetime
logging.disable(sys.maxsize)
warnings.filterwarnings('ignore')

# Detect hardware, return appropriate distribution strategy, if tpus are available tpus are used
# Else if gpus are available gpus are used. If neither are available computation is done with CPUs
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None
if tpu:
    # Distribution strategy if tpus are available and is to be used
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print('Using {} TPUs'.format(strategy.num_replicas_in_sync))
elif tf.config.list_physical_devices('GPU'):
    # Distribution strategy in case of multiple GPUs
    strategy = tf.distribute.MirroredStrategy()
    print('Using {} GPUs'.format(strategy.num_replicas_in_sync))
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.MirroredStrategy()
    print('No GPU nor TPU. Running on CPU')
AUTOTUNE = tf.data.experimental.AUTOTUNE


NUM_ACTION = 6
NUM_OBJECT = 14
NUM_LOCATION = 4

train_df = pd.read_csv(FOLDER_PATH+'train_data.csv')
val_df = pd.read_csv(FOLDER_PATH+'valid_data.csv')
action_enc = LabelEncoder()
action_train = action_enc.fit_transform(train_df['action'])
action_val = action_enc.transform(val_df['action'])
object_enc = LabelEncoder()
object_train = object_enc.fit_transform(train_df['object'])
object_val = object_enc.transform(val_df['object'])
location_enc = LabelEncoder()
location_train = location_enc.fit_transform(train_df['location'])
location_val = location_enc.transform(val_df['location'])
texts = train_df['transcription'].values
texts = list(texts)
val_texts = val_df['transcription'].values
val_texts = list(val_texts)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
train_data = tokenizer(texts, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='tf')
val_data = tokenizer(val_texts, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='tf')


y_train = {'action': action_train, 'object':object_train, 'location': location_train}
y_val = {'action': action_val, 'object':object_val, 'location': location_val}
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_data), y_train)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_data), y_val)).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32, name='input_ids')
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32, name='attention_mask')
    bert_model = TFRobertaModel.from_pretrained("roberta-base")
    x = bert_model(ids,attention_mask=att)
    x1 = tf.keras.layers.Flatten()(x[1])
    x1 = tf.keras.layers.Dense(NUM_ACTION, name='action')(x1)

    x2 = tf.keras.layers.Flatten()(x[1])
    x2 = tf.keras.layers.Dense(NUM_OBJECT, name='object')(x2)

    x3 = tf.keras.layers.Flatten()(x[1])
    x3 = tf.keras.layers.Dense(NUM_LOCATION, name='location')(x3)
    model = tf.keras.models.Model(inputs=[ids, att], outputs=[x1,x2,x3])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy(),tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='Top_3')],
        )
    return model
with strategy.scope():
    model = build_model()

csv_logger = CSVLogger(FOLDER_PATH+'log.csv', append=True, separator=';')
history=model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1, callbacks=[csv_logger])

model.save_weights(FOLDER_PATH+'model.h5')
np.save(FOLDER_PATH+'action_encoder.npy', action_enc.classes_)
np.save(FOLDER_PATH+'object_encoder.npy', object_enc.classes_)
np.save(FOLDER_PATH+'location_encoder.npy', location_enc.classes_)