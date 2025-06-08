# Image-description-generation

## Participants:  
  
Polina O - People detection and gender/age/amount classification   
Egor Sokolov - Image description  
Artyom Fedotov - Interface elements detection  
Ilya Gerasimov - Text recognition  

## Project description
Проект представляет собой микросервисную систему для генерации описаний изображений.

Система состоит из нескольких микросервисов, каждый из которых отвечает за свою часть обработки изображения:

1. human-detection: детекция людей, определение пола, возраста, количества
2. image-captioning: генерация общего описания изображения
3. text-detection: распознавание текста на изображении
4. ui-detection: детекция элементов пользовательского интерфейса (UI)
5. web-api: основной API шлюз, собирающий результаты всех микросервисов, объединяющий их и (опционально) передающий в LLM для генерации финального текста


### Как предлагаю работать:  
Для разработки каждому участнику рекомендуется создать отдельную ветку, например:

- feature/human-detection

- feature/image-captioning

- feature/text-detection

- feature/ui-detection

Каждый участник работает в своей папке в services, в которой:

* app.py — основной код FastAPI сервиса

* requirements.txt — зависимости (можно добавлять свои по задаче)

* DOCKERFILE — сборка контейнера

* .gitignore — файлы и папки, которые не будут загружаться в репозиторий

* data - различные данные (датасеты и тп)

* Остальные папки создаются вами

Для работы вы переходите в свою папку (cd <название папки>), 
создаёте виртуально окружение (
    """python -m venv venv"""
    """source venv/bin/activate"""
)

И так работаете в своей ветке.  
Таким образом, у нас будет единая архитектура и быстрый доступ к работам друг друга для нас и для куратора.