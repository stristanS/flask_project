Порядок действий:

- ### Список доступных для обучения моделей можно посмотреть через: requests.get('http://127.0.0.1:8080/post_data')<br />
- ### Загрузка данных происходит через requests.post('http://127.0.0.1:8080/post_data', json = data)<br />
    - При этом передаваемый json имеет следующий формат: {payload: файл.json, target_col_name: str, columns_to_drop: list/None')<br />
        - payload - данные в формате .json для обучения модели. Обязательное поле<br />
        - target_col_name - название колонки таргета, тип string. Обязательное поле<br />
        - columns_to_drop - колонки, которые следует удалить, list или None.<br />
- ### Обучение модели происходит через requests.post('http://127.0.0.1:8080/train_model/<id модели>', json = data)<br />
    - <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 (регрессия) или 2 (классификация).<br />
    - При этом передаваемый json - это параметры модели в формате н-р: {'fit_intercept': True. В качестве параметров можно использовать названия параметров для моделей из п.0 из библиотеки sklearn, соблюдая нейминг. Можно не передавать параметры, тогда обучение происходит с дефолтными.<br />
- ### Предсказание модели происходит через requests.post('http://127.0.0.1:8080/predict/<id модели>', json = data)<br />
    - <id модели> должен совпадать с указанным при обучении <id модели>.<br />
    - Передаваемый json это данные в формате .json для тестирования модели. Обязательное поле.<br />
- ### Повторное обучение модели происходит через requests.put('http://127.0.0.1:8080/alter/<id модели>', json = data)<br />
    - <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 (регрессия) или 2 (классификация). Сначала нужно обучить исходную модель из п.2.<br />
    - Передаваемый json имеет следующий формат:{payload: файл.json, target_col_name: str, columns_to_drop: list/None, params: {}')<br />
        - payload - данные в формате .json для обучения модели. Обязательное поле<br />
        - target_col_name - название колонки таргета, тип string. Обязательное поле<br />
        - columns_to_drop - колонки, которые следует удалить, list или None.<br />
        - params - параметры, с которыми будет переобучаться модель, dict.<br />
- ### Удаление модели происходит через requests.delete('http://127.0.0.1:8080/alter/1<id модели>')<br />
    - <id модели> в пути означает индекс модели из списка 0, т.е. возможные значения 1 или 2. Сначала нужно обучить исходную модель из п.2<br />
- ### Расчет метрики качества: 
    - сначала необходимо обучить модель и сделать на ней предсказания.
    - также нужно сформировать collection в БД для метрики качества запустив requests.get('http://127.0.0.1:8080/create_metric/<id_модели>') 
    - затем можно запустить расчет метрики качества requests.post('http://127.0.0.1:8080/metric/<id_модели>', json= data), где data в формате {'payload': payload},а  payload - список значений таргет переменной.
    - в ответ на запрос предыдущего пункта возвращается task_id, т.к расчет выполняет worker
    - чтобы посмотреть результат работы worker: requests.post('http://127.0.0.1:8080/results/<task_id>))
- ### Вывод метрики качества из БД: requests.get('http://127.0.0.1:8080/get_metric/<id_модели>')
- ### Удалить метрику из БД: requests.get('http://127.0.0.1:8080/delete_metric/<id_модели>')


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
    
