from absl import app
from absl import flags
from absl import logging
from functools import partial
from functools import wraps
import numpy as np
import queue
import sounddevice as sd
import threading
import whisper
from time import time as now
import configparser
import soundfile as sf

FLAGS = flags.FLAGS

# Определение флагов с комментариями
flags.DEFINE_string('model_name', 'base.en', 'Версия модели OpenAI Whisper для использования.')
flags.DEFINE_string('language', 'en', 'Язык для использования или пусто для автоопределения.')
flags.DEFINE_string('input_device', 'default', 'Устройство ввода для записи аудио.')

flags.DEFINE_integer('sample_rate', 16000, 'Частота дискретизации записываемого аудио.')
flags.DEFINE_integer('num_channels', 1, 'Количество каналов записываемого аудио.')
flags.DEFINE_integer('channel_index', 0, 'Индекс канала для транскрипции.')
flags.DEFINE_integer('chunk_seconds', 10, 'Длина в секундах каждого фрагмента записанного аудио.')
flags.DEFINE_string('latency', 'low', 'Задержка потока записи.')

flags.DEFINE_float('confidence_threshold', 0.5, 'Порог уверенности для транскрипции Whisper.')

# Путь к файлу настроек
flags.DEFINE_string('settings_file', 'settings.ini', 'Путь к файлу настроек.')

# Декоратор для измерения времени выполнения функций
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = now()
        result = func(*args, **kwargs)
        stop = now()
        logging.debug(f'{func.__name__} заняла {stop-start:.3f} сек.')
        return result
    return wrapper

# Функция для транскрипции аудио
@timed
def transcribe(model, audio, confidence_threshold):
    # Запуск модели Whisper для транскрипции аудио.
    result = whisper.transcribe(model=model, audio=audio)

    # Использование транскрибированного текста, если уверенность превышает порог.
    confidence = result.get('confidence', 0.0)
    if confidence >= confidence_threshold:
        text = result['text'].strip()
        logging.info(text)

# Функция для обработки аудио
def process_audio(audio_queue, model, confidence_threshold):
    while True:
        # Блокировка до появления следующего фрагмента аудио в очереди.
        audio = audio_queue.get()

        # Транскрипция последнего фрагмента аудио.
        transcribe(model=model, audio=audio, confidence_threshold=confidence_threshold)

# Глобальные переменные для управления записью аудио
recording_chunk = None
recording_start_time = 0
user_speaking = False
queue_audio = queue.Queue()  # Добавим глобальную очередь для передачи в record_audio

# Функция для записи аудио
def record_audio(callback, block_size, queue_audio, **kwargs):
    global recording_chunk, recording_start_time, user_speaking

    with sd.InputStream(callback=callback, blocksize=block_size, **kwargs):
        recording_chunk = np.zeros(block_size, dtype=np.float32)
        recording_start_time = now()
        user_speaking = False

        print("Поток записи аудио запущен.")

        while True:
            sd.sleep(10)

            if now() - recording_start_time > 10:
                if user_speaking:
                    queue_audio.put(recording_chunk.copy())
                    print("Голос был записан в очередь для распознавания текста.")
                    user_speaking = False
                recording_chunk = np.zeros(block_size, dtype=np.float32)
                recording_start_time = now()

            if recording_chunk is not None:
                volume_level = np.max(np.abs(recording_chunk))
                if volume_level > 0.1:
                    user_speaking = True
                    print("Голос был обнаружен.")

        print("Поток записи аудио завершен.")

# Основная функция
def main(argv):
    # Загрузка настроек из файла конфигурации
    logging.info(f'Загрузка настроек из {FLAGS.settings_file}...')

    # Используем библиотеку configparser для загрузки настроек из файла
    config = configparser.ConfigParser()
    config.read(FLAGS.settings_file)

    # Устанавливаем значения флагов из раздела DEFAULT
    if config.has_section('DEFAULT'):
        for option in config.options('DEFAULT'):
            if hasattr(FLAGS, option):
                # Установка значений, преобразуя типы данных при необходимости
                flag_value = getattr(FLAGS, option)
                if isinstance(flag_value, int):
                    setattr(FLAGS, option, config.getint('DEFAULT', option))
                elif isinstance(flag_value, float):
                    setattr(FLAGS, option, config.getfloat('DEFAULT', option))
                else:
                    setattr(FLAGS, option, config.get('DEFAULT', option))

    # Загрузка модели Whisper в память, скачивание, если необходимо.
    logging.info(f'Загрузка модели "{FLAGS.model_name}"...')
    model = whisper.load_model(name=FLAGS.model_name)

    # Первый запуск модели медленный (инициализация буфера), поэтому запускаем его один раз без аудио.
    logging.info('Подготовка модели...')
    block_size = FLAGS.chunk_seconds * FLAGS.sample_rate
    whisper.transcribe(model=model, audio=np.zeros(block_size, dtype=np.float32))

    # Запуск потока для обработки фрагментов аудио
    audio_queue = queue.Queue()
    audio_processing_thread = threading.Thread(target=process_audio, args=(audio_queue, model, FLAGS.confidence_threshold))
    audio_processing_thread.daemon = True
    audio_processing_thread.start()

    # Запуск потока для записи аудио
    callback = partial(stream_callback, audio_queue=audio_queue)
    recording_thread = threading.Thread(target=record_audio, args=(callback, block_size, queue_audio), kwargs={'samplerate': FLAGS.sample_rate, 'device': FLAGS.input_device, 'channels': FLAGS.num_channels, 'dtype': np.float32, 'latency': FLAGS.latency})

    recording_thread.start()

    print("Программа запущена. Ожидание завершения работы...")

    # Ожидание завершения обоих потоков
    recording_thread.join()
    audio_processing_thread.join()

    print("Программа завершена.")

# Функция обратного вызова для потока записи
def stream_callback(indata, frames, time, status, audio_queue):
    if status:
        logging.warning(f'Статус обратного вызова потока: {status}')

    # Добавление этого фрагмента аудио в очередь.
    audio = indata[:, FLAGS.channel_index].copy()
    audio_queue.put(audio)

if __name__ == '__main__':
    app.run(main)
