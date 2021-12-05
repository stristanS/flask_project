import time
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
from pymongo import MongoClient

from celery import Celery

CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

client = MongoClient('mongodb://mongo:27017/')
db = client['flask_database']
collection = db['flask_collection']


@celery.task(name = 'eval')
def eval(model_id, data):
    """Функция для расчета метрики качества уже обученной модели, на основе которой сделаны предсказания"""
    y_pred = pd.read_csv("../my_app_storage/{name}".format(name = str(model_id)) + '.csv', index_col=0)
    y_pred = list(np.array(y_pred.iloc[:, 0]))
    y = data['payload']
    if model_id == 1:
        metric = mean_squared_error(y_pred, y)
        collection.update_one({'_id': model_id, 'mse': 0}, {"$set": {'mse': metric}})  # UPDATE
        return {'mean_squared_error': metric}
    else:
        metric = roc_auc_score(y_pred, y)
        collection.update_one({'_id': model_id, 'roc_auc': 0}, {"$set": {'roc_auc': metric}}) # UPDATE
        return {'roc_auc_score': metric}




