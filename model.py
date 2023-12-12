import threading
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
import sounddevice as sd
FLAGS = flags.FLAGS

# Определение настроек
file1 = open("settings.ini", "r")
line = file1.readline()
model_file = file1.readline()
language_file = file1.readline()
input_device_file = file1.readline()
confidence_file = file1.readline()
processing_interval_file = file1.readline()
timeout_file = file1.readline()
sample_rate_file = file1.readline()
num_channels_file = file1.readline()
channel_index_file = file1.readline()
chunk_seconds_file = file1.readline()
latency_file = file1.readline()


flags.DEFINE_string('model_name', model_file.strip(),
                    'The version of the OpenAI Whisper model to use.')
flags.DEFINE_string('language', language_file.strip(),
                    'The language to use or empty to auto-detect.')
flags.DEFINE_string('input_device', input_device_file.strip(), 'The input device used to record audio.')

confidence_value = confidence_file.split('=')[-1].strip()
flags.DEFINE_float('confidence', confidence_file.strip(), 'Минимальная уверенность для использования.')

flags.DEFINE_integer('processing_interval', processing_interval_file.strip(), 'Интервал обработки аудио в секундах.')

flags.DEFINE_integer('timeout', timeout_file.strip(), 'Таймаут работы приложения в секундах.')

flags.DEFINE_integer('sample_rate', sample_rate_file.strip(),
                     'The sample rate of the recorded audio.')
flags.DEFINE_integer('num_channels', num_channels_file.strip(),
                     'The number of channels of the recorded audio.')
flags.DEFINE_integer('channel_index', channel_index_file.strip(),
                     'The index of the channel to use for transcription.')
flags.DEFINE_integer('chunk_seconds', chunk_seconds_file.strip(),
                     'The length in seconds of each recorded chunk of audio.')
flags.DEFINE_string('latency', latency_file.strip(), 'The latency of the recording stream.')

THRESHOLD_LEVEL = 0.2  # Примерный порог, подстройте под свои нужды

def check_microphone_level():
    while True:
        # Запись небольшого блока аудио для анализа уровня громкости
        block_size = FLAGS.chunk_seconds * FLAGS.sample_rate
        indata = sd.rec(frames=block_size, channels=FLAGS.num_channels, dtype=np.float32)
        sd.wait()

        # Вычисление уровня громкости (просто пример, может потребоваться другой способ)
        volume_level = np.max(np.abs(indata))

        # Вывод сообщения в консоль
        print(f"Громкость в микрофоне: {volume_level}")



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
        
        check_microphone_thread = threading.Thread(target=check_microphone_level, daemon=True)
        check_microphone_thread.start()    
        while True:
            # Обработка блоков аудио из очереди
            process_audio(audio_queue, model)

if __name__ == '__main__':
    app.run(main)