import requests
import pandas as pd
import json
import numpy as np

"""__________________ ТЕСТ НА ДАННЫХ BOSTON DATASET (РЕГРЕССИЯ)_______________________"""

"""Загрузка данных для обучения"""
payload = pd.read_csv('data/boston_train.csv', index_col = 0).to_json()
data = {'payload':payload, 'target_col_name': 'medv', 'columns_to_drop': None}
response = requests.post('http://127.0.0.1:8080/post_data', json = data, timeout =5)
print(response, response.json())

"""Вывод моделей, доступных для обучения"""
response_1 = requests.get('http://127.0.0.1:8080/post_data')
print(response_1, response_1.json())
#
"""Обучение модели 1"""
model_params = {'fit_intercept': True, 'normalize': True}
response_2 = requests.post('http://127.0.0.1:8080/train_model/1', json = model_params)
print(response_2, response_2.json())
#
"""Предсказание модели 1"""
payload = (pd.read_csv('data/boston_test.csv', index_col = 0).to_json())
response_3 = requests.post('http://127.0.0.1:8080/predict/1', json =payload)
print(response_3, response_3.json())
#

"""CREATE collection для метрики модели 1"""
response_3_2 = requests.get('http://127.0.0.1:8080/create_metric/1')
print(response_3_2, response_3_2.json())
# #
"""UPDATE метрики модели 1"""
df = pd.read_csv('data/boston_test.csv', index_col = 0)
target_col_name = 'medv'
payload = list(np.array(df['medv']))
data = {'payload': payload}
response_3_1 = requests.post('http://127.0.0.1:8080/metric/1', json= data)
print(response_3_1, response_3_1.json())

"""Статус воркера для расчета метрики модели 1"""
response_3_2 = requests.post('http://127.0.0.1:8080/results/3ea4cb7a-15a3-4176-8ea7-0b70272f416c')
print(response_3_2, response_3_2.json())
#
"""GET метрики модели 1"""
response_3_2 = requests.get('http://127.0.0.1:8080/get_metric/1')
print(response_3_2, response_3_2.json())

"""DELETE метрики модели 1, если она уже есть"""
response_3_2 = requests.get('http://127.0.0.1:8080/delete_metric/1')
print(response_3_2, response_3_2.json())

"""Переобучение модели 1"""
payload = (pd.read_csv('data/boston_train.csv', index_col = 0).to_json())
data = {'payload':payload, 'target_col_name': 'medv', 'columns_to_drop': None, 'params': model_params}
response_4 = requests.put('http://127.0.0.1:8080/alter/1', json = data)
print(response_4, response_4.json())

"""Удаление модели 1"""
response_5 = requests.delete('http://127.0.0.1:8080/alter/1')
print(response_5, response_5.json())

"""__________________ ТЕСТ НА ДАННЫХ TITANIC DATASET _______________________"""

"""Загрузка данных для обучения"""
payload = pd.read_csv('data/titanic_train.csv', index_col = 0).to_json()
data = {'payload':payload, 'target_col_name': 'Survived', 'columns_to_drop': ['Name', 'Ticket']}
response = requests.post('http://127.0.0.1:8080/post_data', json = data, timeout =5)
print(response, response.json())

"""Вывод моделей, доступных для обучения"""
response_1 = requests.get('http://127.0.0.1:8080/post_data')
print(response_1, response_1.json())
#
"""Обучение модели 2"""
model_params = {'fit_intercept': True}
response_2 = requests.post('http://127.0.0.1:8080/train_model/2', json = model_params)
print(response_2, response_2.json())
#
"""Предсказание модели 2"""
payload = (pd.read_csv('data/titanic_test.csv', index_col = 0).to_json())
response_3 = requests.post('http://127.0.0.1:8080/predict/2', json =payload)
print(response_3, response_3.json())
#

"""CREATE collection для метрики модели 2"""
response_3_2 = requests.get('http://127.0.0.1:8080/create_metric/2')
print(response_3_2, response_3_2.json())

"""UPDATE метрики модели 2"""
df = pd.read_csv('data/titanic_test.csv', index_col = 0)
payload = list(df['Survived'])
data = {'payload': payload}
response_3_1 = requests.post('http://127.0.0.1:8080/metric/2', json= data)
print(response_3_1, response_3_1.json())

"""Статус воркера для расчета метрики модели 2"""
response_3_2 = requests.post('http://127.0.0.1:8080/results/733e0a14-aa32-49c9-b1cb-28cd71c52c1f')
print(response_3_2, response_3_2.json())

"""GET метрики модели 2"""
response_3_2 = requests.get('http://127.0.0.1:8080/get_metric/2')
print(response_3_2, response_3_2.json())

"""DELETE метрики модели 2, если она уже есть"""
response_3_2 = requests.get('http://127.0.0.1:8080/delete_metric/2')
print(response_3_2, response_3_2.json())

"""Переобучение модели 2"""
payload = (pd.read_csv('data/titanic_train.csv', index_col = 0).to_json())
data = {'payload':payload, 'target_col_name': 'Survived', 'columns_to_drop': ['Name', 'Ticket'], 'params': model_params}
response_4 = requests.put('http://127.0.0.1:8080/alter/2', json = data)
print(response_4, response_4.json())

"""Удаление модели 2"""
response_5 = requests.delete('http://127.0.0.1:8080/alter/2')
print(response_5, response_5.json())