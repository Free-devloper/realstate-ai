
import os
import secrets

from dotenv import load_dotenv

load_dotenv()
basedir = os.path.abspath(os.path.dirname(__file__))


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {'0', 'false', 'no', 'off', ''}


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(16)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    HF_API_TOKEN = os.environ.get('HF_API_TOKEN')
    HF_API_URL = os.environ.get('HF_API_URL') or 'https://router.huggingface.co/v1/chat/completions'
    HF_MODEL = os.environ.get('HF_MODEL') or 'Qwen/Qwen2.5-7B-Instruct'
    HF_PROVIDER = os.environ.get('HF_PROVIDER') or 'auto'
    HF_DISABLED = env_flag('HF_DISABLED', False)
    HF_MODEL_FALLBACKS = os.environ.get('HF_MODEL_FALLBACKS') or ''
    LLAMA_CPP_ENABLED = env_flag('LLAMA_CPP_ENABLED', True)
    LLAMA_CPP_URL = os.environ.get('LLAMA_CPP_URL') or 'http://127.0.0.1:8080/v1/chat/completions'
    LLAMA_CPP_MODEL = os.environ.get('LLAMA_CPP_MODEL') or 'qwen3.5-9b'
    LLAMA_CPP_CONTEXT_WINDOW = int(os.environ.get('LLAMA_CPP_CONTEXT_WINDOW', '40000'))
    AI_REQUEST_TIMEOUT = int(os.environ.get('AI_REQUEST_TIMEOUT', '20'))
    AI_STREAM_CHUNK_DELAY = float(os.environ.get('AI_STREAM_CHUNK_DELAY', '0.05'))
    QUERY_CACHE_ENABLED = env_flag('QUERY_CACHE_ENABLED', True)
    QUERY_CACHE_TTL = int(os.environ.get('QUERY_CACHE_TTL', '180'))
    QUERY_CACHE_MAX_ENTRIES = int(os.environ.get('QUERY_CACHE_MAX_ENTRIES', '256'))
    REAL_ESTATE_API_PROVIDER = os.environ.get('REAL_ESTATE_API_PROVIDER') or 'rentcast'
    RENTCAST_API_KEY = os.environ.get('RENTCAST_API_KEY')
    RENTCAST_BASE_URL = os.environ.get('RENTCAST_BASE_URL') or 'https://api.rentcast.io/v1'
    NOMINATIM_SEARCH_URL = os.environ.get('NOMINATIM_SEARCH_URL') or 'https://nominatim.openstreetmap.org/search'
    APP_PUBLIC_NAME = os.environ.get('APP_PUBLIC_NAME') or 'RealEstateAI/1.0'
