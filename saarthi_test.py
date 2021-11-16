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

test_df = pd.read_csv(TEST_PATH)
test_df.drop('path',axis='columns', inplace=True)
test_df.drop_duplicates(inplace=True)
print("Test shape is", test_df.shape)

action_enc = LabelEncoder()
action_enc.classes_ = np.load(FOLDER_PATH+'action_encoder.npy', allow_pickle=True)
action_test = action_enc.transform(test_df['action'])

object_enc = LabelEncoder()
object_enc.classes_ = np.load(FOLDER_PATH+'object_encoder.npy', allow_pickle=True)
object_test = object_enc.transform(test_df['object'])

location_enc = LabelEncoder()
location_enc.classes_ = np.load(FOLDER_PATH+'location_encoder.npy', allow_pickle=True)
location_test = location_enc.transform(test_df['location'])

test_texts = test_df['transcription'].values
test_texts = list(test_texts)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

MAX_LEN = 13
test_data = tokenizer(test_texts, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='tf')
y_test = {'action': action_test, 'object':object_test, 'location': location_test}
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_data), y_test)).batch(BATCH_SIZE)

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

model.load_weights(FOLDER_PATH+'model.h5')
ans = model.predict(test_dataset)

action_preds = action_enc.inverse_transform(np.argmax(ans[0],axis=1))
object_preds = object_enc.inverse_transform(np.argmax(ans[1],axis=1))
location_preds = location_enc.inverse_transform(np.argmax(ans[2],axis=1))

pred_df = pd.DataFrame({'input':test_df['transcription'],'action':test_df['action'],'object':test_df['object'],'location':test_df['location'],
                        'action_preds':action_preds, 'object_preds':object_preds, 'location_preds':location_preds})

# micro f1 score
action_f1 = f1_score(pred_df['action_preds'],pred_df['action'], average='micro')
object_f1 = f1_score(pred_df['object_preds'],pred_df['object'], average='micro')
location_f1 = f1_score(pred_df['location_preds'],pred_df['location'], average='micro')
print('F1 score for action-->',action_f1)
print('F1 score for object-->',object_f1)
print('F1 score for location-->',location_f1)
pred_df.to_csv(FOLDER_PATH+'predictions.csv',index=False)