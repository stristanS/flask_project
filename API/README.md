Порядок действий:

0. Список доступных для обучения моделей можно посмотреть через: requests.get('{host}/post_data')<br />
1. Загрузка данных происходит через requests.post('{host}/post_data', json = data)<br />
    1.1. При этом передаваемый json имеет следующий формат: {payload: файл.json, target_col_name: str, columns_to_drop: list/None')<br />
        1.1.1. payload - данные в формате .json для обучения модели. Обязательное поле<br />
        1.1.2. target_col_name - название колонки таргета, тип string. Обязательное поле<br />
        1.1.3. columns_to_drop - колонки, которые следует удалить, list или None.<br />
2. Обучение модели происходит через requests.post('{host}/train_model/<id модели>', json = data)<br />
    2.1. <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 (регрессия) или 2 (классификация).<br />
    2.2. При этом передаваемый json - это параметры модели в формате н-р: {'fit_intercept': True. В качестве параметров можно использовать названия параметров для моделей из п.0 из библиотеки sklearn, соблюдая нейминг. Можно не передавать параметры, тогда обучение происходит с дефолтными.<br />
3. Предсказание модели происходит через requests.post('{host}/predict/<id модели>', json = data)<br />
    3.1. <id модели> должен совпадать с указанным при обучении <id модели>.<br />
    3.2. Передаваемый json это данные в формате .json для тестирования модели. Обязательное поле.<br />
4. Повторное обучение модели происходит через requests.put('{host}/alter/<id модели>', json = data)<br />
    4.1. <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 (регрессия) или 2 (классификация). Сначала нужно обучить исходную модель из п.2.<br />
    4.2. Передаваемый json имеет следующий формат:{payload: файл.json, target_col_name: str, columns_to_drop: list/None, params: {}')<br />
        4.2.1. payload - данные в формате .json для обучения модели. Обязательное поле<br />
        4.2.2. target_col_name - название колонки таргета, тип string. Обязательное поле<br />
        4.2.3. columns_to_drop - колонки, которые следует удалить, list или None.<br />
        4.2.4. params - параметры, с которыми будет переобучаться модель, dict.<br />
5. Удаление модели происходит через requests.delete('{host}/alter/1<id модели>')<br />
    5.1. <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 или 2. Сначала нужно обучить исходную модель из п.2<br />

__Для тестирования использовались только датасеты с небольшим количеством наблюдений (до 1000) и признаков.__

Пример праметров для BOSTON DATASET:

    1. payload: 
        1.1. train: pd.read_csv('data/boston_train.csv', index_col = 0).to_json()
        1.2. test: pd.read_csv('data/boston_test.csv', index_col = 0).to_json()
    2. target_col_name: 'medv' 
    3. columns_to_drop: None / любая
    4. <id модели>: 1 
    5. model_params = None / {fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=False}


Пример праметров для TITANIC DATASET:
    
    1. payload: 
        1.1.  train: pd.read_csv('data/titanic_train.csv', index_col = 0).to_json()
        1.2. test: pd.read_csv('data/titanic_test.csv', index_col = 0).to_json()
    3. target_col_name: 'Survived' 
    4. columns_to_drop: None / ['Name', 'Ticket'] (желательно удалить колонки, иначе большое признаковое пространство) 
    5. <id модели>: 2
    6. model_params = None / {'penalty': 'l2', etc}, но строго параметры sklearn.linear_model.LogisticRegression Parameters 
    
