Структура проекта.

    .
    ├── API
    │   ├── data          		        # .csv файлы для обучения/предсказания
    │   ├── api.py    			# flask - приложение для загрузки данных, обучения и предсказания
    │   ├── Dockerfile    	
    │   ├── example_http_requests.py  	# тестирование работы всех методов flask - приложения и workera
    │   ├── README.md    		
    │   └── requirements.txt             
    ├── my_app_storage			
    ├── worker				
    │   ├── Dockerfile           		
    │   ├── requirements.txt   
    │   ├── task.py  			# расчет метрик качества обученных моделей (вычисления не в рамках Flask-приложения)
    └── README.md


1. Работа с БД: 
	- В качестве БД использовался docker образ mongo.
	- Реализован CRUD только для расчета метрики качества. Реализация представлена в методах класса MLModelsDAO: create_metric, get_metric, delete_metric; Update реализован в рамках отдельного воркера worker/task.py.

2. Вычисления не в рамках Flask-приложения:
	- реализованы с помощью celery и redis

3. Ссылки на Docker образы flask приложения и workera:
	- https://hub.docker.com/repository/docker/ttris117/web_app
	- https://hub.docker.com/repository/docker/ttris117/worker

4. Запуск всех контейнеров через:
	- docker-compose build
	- docker-compose up

5. Протестировать можно запуском файла example_http_requests.py
