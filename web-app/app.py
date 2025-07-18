import streamlit as st
import requests
from PIL import Image

# Настройки API
CAPTION_API_URL = 'http://caption_api:8000/caption'
TTS_API_URL = 'http://tts_api:8002/tts'


# Оформление страницы 
st.set_page_config(
    page_title="Image Captioning Project",
    page_icon="📸",
    layout="centered",
)

# Заголовок проекта
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Image Captioning Project</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 10px;'>
    <a href='https://github.com/CriminalSparrow/Image-description-generation' target='_blank' style='color: #0366d6; text-decoration: none; font-size: 18px;'>
        🔗 Ссылка на GitHub проекта
    </a>
</div>
""", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: gray;'>Создание и перевод описаний для изображений</h4>", unsafe_allow_html=True)
st.markdown("---")

# Загрузка файла  
uploaded_file = st.file_uploader("Загрузите изображение (поддерживаются форматы PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'])

# Выбор степени детализации
detail_level = st.selectbox(
    'Выберите степень детализации описания:',
    ['Краткое', 'Обычное', 'Подробное']
)

# Сопоставление выбора с prompt
prompt_mapping = {
    'Краткое': '<CAPTION>',
    'Обычное': '<DETAILED_CAPTION>',
    'Подробное': '<MORE_DETAILED_CAPTION>'
}

selected_prompt = prompt_mapping[detail_level]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_container_width=True)

    if st.button('Сгенерировать описание'):
        with st.spinner('Внимательно изучаю изображение...'):
            files = {'file': uploaded_file.getvalue()}
            params = {'prompt': selected_prompt}
            response = requests.post(CAPTION_API_URL, files=files, params=params)
            
            if response.status_code == 200:
                data = response.json()
                caption_ru = data['caption']
                confidence = data['mean_confidence']

                st.markdown(f"### Сгенерированное описание:")
                st.success(f"{caption_ru}")

                st.markdown(f"### Средняя уверенность модели:")
                st.info(f"{confidence * 100:.2f}%")

                # Генерация TTS
                with st.spinner('Генерация аудио...'):
                    tts_payload = {
                        "text": caption_ru,
                        "lang": "ru-RU",
                        "voice": "ru-RU-DmitryNeural"
                    }

                    tts_response = requests.post(TTS_API_URL, json=tts_payload)

                    if tts_response.status_code == 200:
                        st.markdown("### Озвучка описания:")
                        st.audio(tts_response.content, format="audio/mp3")
                    else:
                        st.error('Ошибка генерации аудио')
            else:
                st.error('Ошибка при получении описания')

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        <h5>Проект выполнен:</h5>
        <ul style='list-style-position: inside; text-align: left; display: inline-block;'>
            <li><strong>Polina O</strong> — People detection and gender/age/amount classification</li>
            <li><strong>Egor Sokolov</strong> — Image description</li>
            <li><strong>Artyom Fedotov</strong> — Interface elements detection</li>
            <li><strong>Ilya Gerasimov</strong> — Text recognition</li>
        </ul>
        <p style='color: gray; font-size: 14px;'>2025</p>
    </div>
""", unsafe_allow_html=True)

