from flask import Flask
from flask_restx import Api, Resource
from flask import request, jsonify
import os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
from sklearn import preprocessing
from celery import Celery
from pymongo import MongoClient

CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

app = Flask(__name__)
api = Api(app)

client = MongoClient('mongodb://mongo:27017/')
db = client['flask_database']
collection = db['flask_collection']


class MLModelsDAO:
    def __init__(self):
        self.model_file_name = "../my_app_storage/"
        self.data = None
        self.target_col_name = None
        self.columns_to_drop = None
        self.train_x_ohe = None
        self.models = [{'model_id': 1, 'model': LinearRegression()}, {'model_id': 2, 'model': LogisticRegression(solver='liblinear')}]

    def get(self):
        models_dict = {}
        for model in self.models:
            models_dict[model['model_id']] = str(model['model'])
        return models_dict

    def post(self, data):
        try:
            self.data = pd.read_json(data['payload'])
            self.target_col_name = data['target_col_name']
            self.columns_to_drop = data['columns_to_drop']
            if self.target_col_name not in self.data.columns:
                api.abort(400, 'No such column {} in dataframe.'.format(self.target_col_name))
            if self.columns_to_drop is not None:
                for col in self.columns_to_drop:
                    if col not in self.data:
                        api.abort(400, 'No such column {} in dataframe.'.format(col))
        except KeyError:
            api.abort(400, 'Input format allowed: {payload: .json, target_col_name: str, columns_to_drop: list/None')
        except MemoryError:
            api.abort(400, 'MemoryError, unable to allocate data')

    def data_preprocessing(self, data):
        try:
            if self.columns_to_drop is not None:
                data.drop(self.columns_to_drop, axis=1, inplace=True)
            x, y = data.loc[:, data.columns != self.target_col_name], data[self.target_col_name]
            cat_features = []
            for col in x.columns:
                if x[col].dtypes == 'O':
                    cat_features.append(col)
                    x[col].fillna('unseen_cat', inplace=True)
                else:
                    x[col].fillna(-1, inplace=True)
            x = pd.get_dummies(x, columns=cat_features)
            if y.dtypes == 'O':
                le = preprocessing.LabelEncoder()
                y = le.fit_transform(y)
            # x, y = data.loc[:, data.columns != self.target_col_name], data[self.target_col_name]
            return x, y
        except KeyError:
            api.abort(400, 'Invalid input data.')

    def fit(self, model_id, params):
        if self.data is None:
            api.abort(400, 'Please provide dataframe before fit the model')
        x, y = self.data_preprocessing(self.data)
        self.train_x_ohe = x
        for mod in self.models:
            if mod['model_id'] == model_id:
                try:
                    if params:
                        model = mod['model'].set_params(**params)
                    else:
                        model = mod['model']
                    model.fit(x, y)
                    pickle.dump(model, open(self.model_file_name+"{name}.pickle".format(name=model_id), "wb"))
                except ValueError:
                    api.abort(400, 'Invalid parameter for estimator {}.'.format(str(mod['model'])))

    def predict(self, model_id, data):
        data = pd.read_json(data)
        if self.data is None:
            api.abort(400, 'Please provide dataframe and fit the model before prediction.')
        x, _ = self.data_preprocessing(data)
        ##################################
        _, x = self.train_x_ohe.align(x, join='left', axis=1)
        x.fillna(0, inplace=True)
        ###############################
        for model in self.models:
            if model['model_id'] == model_id:
                try:
                    path = os.path.join(self.model_file_name, str(model_id) + '.pickle')
                    model = pickle.load(open(path, 'rb'))
                    print(x)
                    prediction = list(model.predict(x))
                    pd.DataFrame(prediction).to_csv(self.model_file_name+'{name}.csv'.format(name=model_id), index=True)
                    return {'prediction': [int(x) for x in prediction]}
                except FileNotFoundError:
                    api.abort(400, 'Model {} should be fitted first.'.format(str(model['model'])))
                except ValueError:
                    api.abort(400, 'Dimension mismatch. Check provided data features.')

    def retrain(self, model_id, data):
        try:
            params = data['params']
        except KeyError:
            params = None
        self.post(data)
        for model in self.models:
            if model['model_id'] == model_id:
                path = os.path.join(self.model_file_name, str(model_id) + '.pickle')
                if os.path.isfile(path):
                    self.fit(model_id, params)
                else:
                    api.abort(400, 'Train initial model with id {} first.'.format(model_id))

    def delete(self, model_id):
        try:
            path = os.path.join(self.model_file_name, str(model_id) + '.pickle')
            os.remove(path)
        except FileNotFoundError:
            api.abort(400, 'No model for model_id {} found.'.format(model_id))

    def create_metric(self, model_id):
        if model_id == 1:
            collection.insert_one({"_id": model_id, "mse": 0})
        else:
            collection.insert_one({"_id": model_id, "roc_auc": 0})
        return collection.find_one({'_id': model_id})

    def get_metric(self, model_id):
        return collection.find_one({'_id': model_id})

    def delete_metric(self, model_id):
        collection.delete_one({'_id': model_id})


ml_models = MLModelsDAO()


@api.route('/post_data')
class MLModels(Resource):
    """Вывод доступных для обучения моделей и загрузка данных от пользователя"""
    def get(self):
        return ml_models.get()

    def post(self):
        uploaded_file = request.json
        ml_models.post(uploaded_file)
        return jsonify(reply='Data loaded')


@api.route('/train_model/<int:model_id>')
class MLModels(Resource):
    """Загрузка данных и обучение модели с учетом заданных параметров"""
    def post(self, model_id):
        params = request.json
        ml_models.fit(model_id, params)
        return jsonify(reply='Train finished')


@api.route('/predict/<int:model_id>')
class MLModels(Resource):
    """Предсказание конкретной модели на данных от пользователя"""
    def post(self, model_id):
        data_for_prediction = request.json
        prediction = ml_models.predict(model_id, data_for_prediction)
        return jsonify(prediction)


@api.route('/alter/<int:model_id>')
class MLModels(Resource):
    """Обучение заново и удаление старой модели"""
    def put(self, model_id):
        params = request.json
        ml_models.retrain(model_id, params)
        return jsonify(reply='Retrain finished')

    def delete(self, model_id):
        ml_models.delete(model_id)
        return jsonify(reply='Model was deleted')


@api.route('/metric/<int:model_id>')
class MLModels(Resource):
    """Расчет метрики качества отдельным workerom"""
    def post(self, model_id):
        uploaded_file = request.json
        task = celery.send_task('eval', args = [model_id, uploaded_file])
        return f'Task_id = {task.id}', 200


@api.route('/results/<string:task_id>')
class MLModels(Resource):
    """Результат расчета метрики качества"""
    def post(self, task_id):
        res = celery.AsyncResult(task_id)
        if res.state == 'PENDING':
            return str(res.state)
        else:
            return str(res.result)

@api.route('/create_metric/<int:model_id>')
class MLModels(Resource):
    """CREATE в БД метрики качества"""
    def get(self, model_id):
        task = ml_models.create_metric(model_id)
        return jsonify(task)


@api.route('/get_metric/<int:model_id>')
class MLModels(Resource):
    """READ из БД метрики качества"""
    def get(self, model_id):
        task = ml_models.get_metric(model_id)
        return jsonify(task)

@api.route('/delete_metric/<int:model_id>')
class MLModels(Resource):
    """DELETE из БД метрики качества"""
    def get(self, model_id):
        ml_models.delete_metric(model_id)
        return jsonify(reply='Metric was deleted from DB')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')
