from absl import app
from absl import flags
from absl import logging
from functools import partial
from functools import wraps
import numpy as np
import queue
import sounddevice as sd
from time import time as now
import whisper
import configparser

FLAGS = flags.FLAGS

# Определение настроек
file1 = open("settings.ini", "r")
line = file1.readline()
model_file = file1.readline()
language_file = file1.readline()
confidence_file = file1.readline()
processing_interval_file = file1.readline()
timeout_file = file1.readline()

# Путь к файлу настроек
flags.DEFINE_string('settings_file', 'settings.ini', 'Путь к файлу настроек.')

flags.DEFINE_string('model_name', model_file.strip(),
                    'The version of the OpenAI Whisper model to use.')
flags.DEFINE_string('language', language_file.strip(),
                    'The language to use or empty to auto-detect.')
flags.DEFINE_string('input_device', 'default', 'The input device used to record audio.')

confidence_value = confidence_file.split('=')[-1].strip()
flags.DEFINE_float('confidence', float(confidence_value), 'Минимальная уверенность для использования.')

flags.DEFINE_integer('processing_interval', processing_interval_file.strip(), 'Интервал обработки аудио в секундах.')

flags.DEFINE_integer('timeout', timeout_file.strip(), 'Таймаут работы приложения в секундах.')

flags.DEFINE_integer('sample_rate', 16000,
                     'The sample rate of the recorded audio.')
flags.DEFINE_integer('num_channels', 1,
                     'The number of channels of the recorded audio.')
flags.DEFINE_integer('channel_index', 0,
                     'The index of the channel to use for transcription.')
flags.DEFINE_integer('chunk_seconds', 10,
                     'The length in seconds of each recorded chunk of audio.')
flags.DEFINE_string('latency', 'low', 'The latency of the recording stream.')


# A decorator to log the timing of performance-critical functions.
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = now()
        result = func(*args, **kwargs)
        stop = now()
        logging.debug(f'{func.__name__} took {stop-start:.3f}s')
        return result
    return wrapper


@timed
def transcribe(model, audio):
    # Run the Whisper model to transcribe the audio chunk.
    result = whisper.transcribe(model=model, audio=audio)

    # Use the transcribed text.
    text = result['text'].strip()
    logging.info(text)


@timed
def stream_callback(indata, frames, time, status, audio_queue):
    if status:
        logging.warning(f'Stream callback status: {status}')

    # Add this chunk of audio to the queue.
    audio = indata[:, FLAGS.channel_index].copy()
    audio_queue.put(audio)


@timed
def process_audio(audio_queue, model):
    # Block until the next chunk of audio is available on the queue.
    audio = audio_queue.get()

    # Transcribe the latest audio chunk.
    transcribe(model=model, audio=audio)


def main(argv):
    # Загрузка настроек из файла конфигурации
    logging.info(f'Loading settings from {FLAGS.settings_file}...')

    # Используем библиотеку configparser для загрузки настроек из файла
    config = configparser.ConfigParser()
   

    # Устанавливаем значения флагов из конфигурационного файла
    for section in config.sections():
        for option in config.options(section):
            if hasattr(FLAGS, option):
                flag_value = getattr(FLAGS, option)
                # Установка значений, преобразуя типы данных при необходимости
                if isinstance(flag_value, int):
                    setattr(FLAGS, option, config.getint(section, option))
                elif isinstance(flag_value, float):
                    setattr(FLAGS, option, config.getfloat(section, option))
                else:
                    setattr(FLAGS, option, config.get(section, option))

    # Загрузка модели Whisper
    logging.info(f'Loading model "{FLAGS.model_name}"...')
    model = whisper.load_model(name=FLAGS.model_name)

    # Предварительный прогон модели (warm-up)
    logging.info('Warming model up...')
    block_size = FLAGS.chunk_seconds * FLAGS.sample_rate
    whisper.transcribe(model=model,
                       audio=np.zeros(block_size, dtype=np.float32))

    # Начало потока аудио и обработка
    logging.info('Starting stream...')
    audio_queue = queue.Queue()
    callback = partial(stream_callback, audio_queue=audio_queue)
    with sd.InputStream(samplerate=FLAGS.sample_rate,
                        blocksize=block_size,
                        device=FLAGS.input_device,
                        channels=FLAGS.num_channels,
                        dtype=np.float32,
                        latency=FLAGS.latency,
                        callback=callback):
        while True:
            # Обработка блоков аудио из очереди
            process_audio(audio_queue, model)

if __name__ == '__main__':
    app.run(main)