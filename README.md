# Image-description-generation

## Participants:  
  
Polina O - People detection and gender/age/amount classification   
Egor Sokolov - Image description  
Artyom Fedotov - Interface elements detection  
Ilya Gerasimov - Text recognition  

## Project description  
Замысел проекта представляет собой микросервисную систему для генерации описаний изображений для слабовидящих пользователей.  
  
Система должна состоять из нескольких микросервисов, каждый из которых отвечает за свою часть обработки изображения:  
  
1. human-detection: детекция людей, определение пола, возраста, количества  
2. image-captioning: генерация общего описания изображения  
3. text-detection: распознавание текста на изображении  
4. ui-detection: детекция элементов пользовательского интерфейса (UI)  
5. web-api: основной API шлюз, собирающий результаты всех микросервисов и объединяющий их.  

В данной ветке был реализован image-captioning.  

## Структура проекта:  

IMAGE-DESCRIPTION-GENERATION/  
│  
├── docker-compose.yml            
├── README.md               
│  
├── services/  
│   ├── image-captioning/       # Сервис генерации описаний  
│   │   ├── Dockerfile  
│   │   ├── app.py  
│   │   └── requirements.txt  
│   │  
│   ├── translator/             # Сервис перевода текста (через OpenRouter API)  
│   │   ├── Dockerfile  
│   │   ├── app.py  
│   │   ├── requirements.txt  
│   │   └── .env                # Ключ для OpenRouter  
│   │  
│   ├── tts/                    # Сервис озвучки текста (Edge TTS)  
│   │   ├── Dockerfile   
│   │   ├── app.py   
│   │   └── requirements.txt   
│  
├── streamlit-app/              # Веб-интерфейс пользователя на streamlit  
│   ├── Dockerfile  
│   ├── app.py   
│   └── requirements.txt  
  
## В данной ветке проекта было реализовано:  
1. image-captioning — генерация описаний на английском с Florence 2 Large.  
2. translator — перевод описания с английского на русский с помощью LLM через OpenRouter.  
3. tts — озвучка на русском с помощью Edge TTS.  
4. streamlit-app — фронтенд для загрузки изображения, выбора детализации, просмотра и прослушивания результата.  

## Описание ноутбуков:
Ноутбуки по подбору, тестированию моделей описания изображений:  
1. Blip2_testing - проба Blip2-opt-2.7b на изображении
2. Nanonets_testing - проба Nanonets на нескольких изображениях
3. testing_models_MAIN - тестирование всех рассмотренных моделей на изображениях, выводы по качеству описаний и производительности.
Рассмотренные модели:  
Blip2-OPT-2.7b, 
Blip large (0.47b)  
Florence 2 large (0.77b)  
git large  
instructblip-vicuna-7b квантованная  
Nanonets  

**Наилучшим решением оказался выбор Floreence-2-large модели. Она сочетает в себе дополнительный функционал для работы с изображениями, быстрый инференс, точные и полные описания. Но, так как модель выдаёт описания только на английском языке, нужно встраивание переводчика в пайплайн для русскоязычных пользователей.** 

4. models_clip_score_evaluation - для формального обоснования выбора модели была посчитана метрика CLIP score для Blip large и Florence 2 large.   
**Florence 2 large показала лучшее качество краткого описания изображений по метрике CLIP score.**  

5. testing-florence-2-large-for-weaknesses - проверка выбранной модели на сложных, шумных изображениях. 
Попытки формально определить изображения, на которых модель может галлюцинировать:
1. Слишком низкая или высокая дисперсия пикселей 
2. Слишком низкий CLIP score
3. Слишком низкая уверенность модели в выданных эмбеддингах: perplexity, mean_probabilities.

**В качестве наиболее обоснованного варианта была выбрана метрика mean_probabilities с порогом 0.64**


## Запуск проекта:  
1. Клонируйте репозиторий:  
git clone https://github.com/CriminalSparrow/Image-description-generation.git  
cd project  

2. Убедитесь, что установлен Docker и Docker Compose.  

3. Соберите проект:  
docker compose up --build  

4. Откройте веб интерфейс по адресу  
http://localhost:8501  
