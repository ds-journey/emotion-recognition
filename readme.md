### Определение человеческих эмоций по данным с камер видеонаблюдений
Авторы проекта: Кубракова Екатерина Александровна, Земскова Мария Викторовна, Анисимов Юрий Сергеевич, Ченчак Михаил Андреевич
<hr/>

Задача проекта — улучшить опыт от посещения музеев, городов или новых мест с помощью информационных технологий.

Цель проекта — сделать посещение людьми музеев, городов или новых мест интересным, информативным и комфортным.

#### Предлагаемое решение

Для реализации цели предлагается приложение по распознаванию эмоций
с помощью модели машинного обучения (нейронной сети).  

Приложение по распознаванию эмоций может быть полезно в музеях по
нескольким причинам.  

Например, анализируя, какие эмоции получают посетители от тех или иных
экспонатов, музей может адаптировать свои
коллекции таким образом, чтобы добиться желаемого отклика.  

Кроме того, посетителям можно будет предоставить персонализированные
рекомендации, на основе их эмоциональных предпочтений.  

Обучение нейронной сети выполнено на основе [набора данных Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset).  

Для запуска [jupyter-ноутбука](./hackaton.ipynb), выполняющего обучение
модели, его необходимо скачать и распаковать.  

Обученные модели сохраняются, начиная с 15й эпохи, в папку `~/data/torch`,
её необходимо создать заранее.  

Уже обученную таким образом модель можно скачать
[здесь](https://drive.google.com/file/d/1cZaV8ab__-jepbpBx2gtjUWeRXeEn-a6/view).

Цель приложения - улучшить опыт посещения музея, повысить уровень удовлетворенности посетителей и помочь музею лучше понять потребности своей аудитории.

С помощью приложения можно решить задачи:

1. Повышение уровня удовлетворенности посетителей: Приложение позволит посетителям выразить свои эмоции и оценки произведений искусства, что позволит музею получить обратную связь от посетителей и улучшить свои услуги.

2. Улучшение интерактивности: Приложение может предложить интерактивные экспозиции, которые реагируют на эмоции посетителей, делая посещение музея более увлекательным и запоминающимся.

3. Персонализация опыта: Приложение может помочь музею адаптировать предложения и экспозиции под конкретные эмоциональные потребности посетителей, что повысит уровень удовлетворенности и комфорта посещения.

4. Анализ данных: Собранные данные об эмоциях посетителей помогут музею лучше понять, какие экспонаты и мероприятия вызывают наибольший интерес, что позволит оптимизировать экспозицию и программу мероприятий.

#### Архитектура

Приложения состоит из двух частей: frontend(приложение на react.js) и backend(серверная часть, разработанная с использованием fastAPI). Части общаются между собой посредством WebSocket.
Frontend отправляет захваченное изображение в backend и получает в ответ оценку "предсказания" эмоции.

Ответ бэкэнда состоит из двух ключей "predictions" и "emotion". Ключ "prediction" состоит из вероятностей "предсказания" каждой эмоции, которые используются для изменения значений шкалы эмоций с каждым кадром. Ключ "emotion" содержит доминирующую эмоцию, которая используется для динамического изменения цвета границы, определенной вокруг человеческого лица. Область вокруг лица определяется с помощью модели TensorflowJS.

#### Используемые технологии и библиотеки
1. ReactJS.
    - Использован для разработки UI.
2. FastAPI.
    - Фреймворк для быстрой разработки серверной части проекта.
3. Tensorflow Keras.
   - Модель возвращаеет вероятностною оценку определения эмоций.
4. Tensorflow JS.
   - Обнаружение лиц на изображении с помощью архитектуры Single Shot Detector с пользовательским модели Blazeface.

#### Запуск проекта

```bash
# установка зависимостей
$ cd server
$ pip install -r requirements.txt
```

```bash
# запуск серверной части
$ cd server
$ export MODEL_FILE=<your_model_file> # путь к файлу модели. Например, в системе Linux так: export MODEL_FILE=/some/path/to/your/model_file.th.
$ uvicorn main:app --reload
```

```bash
# запуск фронтенда
$ cd frontend
$ npm install
$ npm start
```

