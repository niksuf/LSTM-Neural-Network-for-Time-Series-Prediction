# Скрипт для сбора и обработки данных с использованием API Binance и прогнозирования временных рядов с помощью LSTM-модели

## Описание проекта
Цель: разработать скрипт для сбора и обработки данных с использованием API Binance и прогнозирования временных рядов с помощью LSTM-модели.

Приложение выполняет следующие действия:
1. При помощи RESTful API Binance (https://binance-docs.github.io/apidocs/spot/en/) с использованием библиотеки Requests получаем данные о котировках криптовалют. Полученные данные сохраняются в формате ```*.csv```.
2. Выполяется предварительная обработка данных (масштабирование).
3. Разделяет данные на обучающую и тестовую выборки.
4. При помощи TensorFlow и Keras создается и обучается LSTM-модель.
5. Выполняется предсказание следующей цены.
6. Оценка качества модели на тестовой выборке, используя подходящие метрики, такие как среднеквадратичная ошибка (MSE) или средняя абсолютная ошибка (MAE).
7. Визуализация предсказания модели для тестовой выборки на графике.

TODO:
1. Вывести график с предсказанием на час вперед от последних полученных данных.

Часть кода была позаимствована из [статьи](https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks "Статья").
