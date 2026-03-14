import importlib
import sys
from pathlib import Path
from uuid import uuid4

import pytest


MODULES_TO_RESET = [
    'app',
    'config',
    'models',
    'models.agent',
    'models.property',
    'models.user',
    'services.ai_brain',
    'services.agentic_graph',
    'services.realtime_listings',
]


@pytest.fixture()
def app_module(monkeypatch):
    db_path = Path(__file__).resolve().parent / f'test_app_{uuid4().hex}.db'
    monkeypatch.setenv('DATABASE_URL', f"sqlite:///{db_path}")
    monkeypatch.setenv('HF_API_TOKEN', '')
    monkeypatch.setenv('HF_DISABLED', '0')
    monkeypatch.setenv('RENTCAST_API_KEY', '')
    monkeypatch.setenv('LLAMA_CPP_ENABLED', '0')
    monkeypatch.setenv('AI_STREAM_CHUNK_DELAY', '0')

    for module_name in MODULES_TO_RESET:
        sys.modules.pop(module_name, None)

    app_module = importlib.import_module('app')
    app_module.app.config.update(TESTING=True)

    with app_module.app.app_context():
        app_module.db.session.remove()
        app_module.db.drop_all()
        app_module.db.create_all()
        app_module.ensure_seed_data()

    yield app_module

    with app_module.app.app_context():
        app_module.db.session.remove()
        app_module.db.drop_all()
        app_module.db.engine.dispose()

    if db_path.exists():
        db_path.unlink(missing_ok=True)


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()
