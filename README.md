# RealEstate AI

An agentic real estate discovery platform that combines live property retrieval, multi-agent reasoning, conversation memory, and streaming responses in a single chat experience.

## Highlights

- Multi-agent real estate assistant with specialized personas for buyers, investors, and neighborhood guidance
- Shared chat thread where the best-fit agent can respond inline based on the question asked
- LangGraph-powered retrieval and reasoning flow for search, ranking, and response composition
- Live listing retrieval through RentCast, with local inventory fallback when live search is unavailable
- Remote model support through Hugging Face plus local `llama.cpp` fallback
- Streaming responses over Server-Sent Events (SSE)
- Persistent conversation history for logged-in users
- Query routing and short-term response caching for faster repeated lookups
- Dockerized deployment with persistent SQLite storage

## Product Overview

RealEstate AI is built for conversational property discovery. Instead of a static filter form, users can ask natural-language questions like:

- `Find a condo in Sacramento under 650k`
- `Which listing is best for a first-time buyer?`
- `What would be the rental income?`
- `Tell me about the surroundings`

The app decides whether to:

1. answer conversationally,
2. extract search filters,
3. query live listings,
4. rank relevant properties,
5. hand off to the most appropriate agent,
6. and stream the response back into the shared chat.

## Architecture

### Core Stack

- Backend: Flask, SQLAlchemy
- Agent orchestration: LangGraph
- Tool / model integration: LangChain OpenAI-compatible client
- Frontend: HTML, CSS, vanilla JavaScript
- Database: SQLite
- Production server: Gunicorn
- Containerization: Docker, Docker Compose

### Agent Layer

The platform includes multiple specialized agents:

- `Buyer Guide`: fit, tradeoffs, first-time buyer guidance, best-home selection
- `Investment Scout`: yield, rental income, ROI-style reasoning, resale potential
- `Neighborhood Navigator`: surroundings, lifestyle fit, and location-oriented framing

All agents share the same conversation memory and can respond in the same thread. A preferred agent can be selected in the UI, but the system can hand off to another specialist when the question changes.

### Retrieval and Reasoning Flow

The LangGraph pipeline handles:

1. query routing
2. structured argument inference
3. live listing retrieval
4. local inventory fallback
5. ranking and result selection
6. remote model generation or local fallback generation

### Model Fallback Chain

The app supports a resilient model stack:

1. Hugging Face Inference
2. local `llama.cpp` server
3. built-in local reasoning fallback

This makes the chat usable even when a hosted model is unavailable.

## Project Structure

```text
realestateAi/
|-- app.py
|-- config.py
|-- models/
|   |-- agent.py
|   |-- property.py
|   `-- user.py
|-- services/
|   |-- agentic_graph.py
|   |-- ai_brain.py
|   `-- realtime_listings.py
|-- static/
|   |-- script.js
|   `-- style.css
|-- templates/
|   `-- index.html
|-- tests/
|   |-- conftest.py
|   `-- test_app.py
|-- Dockerfile
|-- docker-compose.yml
|-- env-example.txt
`-- requirements.txt
```

## Features

### AI and Agent Experience

- Shared chat with cross-agent replies in one conversation
- Conversation memory stored per logged-in user
- Agent auto-handoff based on user intent
- Streaming and non-streaming response modes
- Query router to skip heavy retrieval for greetings or advisory-only prompts
- Context-aware follow-up handling across turns

### Property Search

- Natural-language search filter inference
- Live listing retrieval from RentCast
- Local sample inventory fallback when live results are unavailable
- Buyer and investor-oriented recommendation ranking
- Favorites, recommendations, and saved conversations

### Developer Experience

- Docker Compose support
- Health checks in Compose
- `.env`-driven configuration
- Automated pytest suite
- Configurable cache, model routing, and AI provider flags

## Requirements

### Local development

- Python 3.10+ recommended
- `pip` or Conda
- Optional: a local `llama.cpp` server running on port `8080`
- Optional: Docker Desktop for containerized runs

### External services

- RentCast API key for live real estate listings
- Hugging Face token if you want hosted model inference

## Environment Variables

Copy `env-example.txt` to `.env` and fill in the values you need.

```env
SECRET_KEY=your-secret-key
DATABASE_URL=
HF_API_TOKEN=
HF_MODEL=
RENTCAST_API_KEY=
HF_PROVIDER=
HF_DISABLED=0
HF_MODEL_FALLBACKS=
LLAMA_CPP_ENABLED=1
LLAMA_CPP_URL=http://127.0.0.1:8080/v1/chat/completions
LLAMA_CPP_MODEL=qwen3.5-9b
LLAMA_CPP_CONTEXT_WINDOW=40000
AI_STREAM_CHUNK_DELAY=0.05
QUERY_CACHE_ENABLED=1
QUERY_CACHE_TTL=180
QUERY_CACHE_MAX_ENTRIES=256
```

### Important settings

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Override the default SQLite database path |
| `RENTCAST_API_KEY` | Enables live listing retrieval |
| `HF_API_TOKEN` | Enables Hugging Face hosted inference |
| `HF_DISABLED` | Skip Hugging Face and go straight to `llama.cpp` |
| `LLAMA_CPP_ENABLED` | Turn local OpenAI-compatible fallback on or off |
| `LLAMA_CPP_URL` | URL for your `llama.cpp` server |
| `AI_STREAM_CHUNK_DELAY` | Controls visible stream speed in the UI |
| `QUERY_CACHE_*` | Tunes query caching behavior |

## Local Setup

### Option 1: Conda

```bash
conda create -n agentic_ai python=3.10 -y
conda activate agentic_ai
pip install -r requirements.txt
```

### Option 2: venv

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Create environment file

```bash
copy env-example.txt .env
```

Update `.env` with your keys and provider choices.

### Run the app

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Docker Setup

The repository includes a ready-to-run Docker setup.

### Start with Docker Compose

```bash
docker compose up --build -d
```

Open:

```text
http://localhost:5002
```

### Docker notes

- Host port `5002` maps to container port `5000`
- SQLite data is persisted in a Docker volume
- The compose file includes a health check against `/api/overview`
- `LLAMA_CPP_URL` is redirected to `http://host.docker.internal:8080/...` inside Docker so the container can reach a host-side `llama.cpp` server

If you do not want to use a local `llama.cpp` server in Docker, disable it in `.env`:

```env
LLAMA_CPP_ENABLED=0
```

## Running `llama.cpp`

If you want local model fallback, run an OpenAI-compatible `llama.cpp` server on port `8080`.

Expected config shape:

- URL: `http://127.0.0.1:8080/v1/chat/completions`
- Model example: `qwen3.5-9b`
- Context window: `40000`

The app can use this for:

- argument extraction
- agent handoff decisions
- remote-response fallback

## API Overview

### Core endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/overview` | `GET` | App and inventory overview |
| `/api/ai/agents` | `GET` | AI agent metadata and runtime status |
| `/api/ai/chat` | `POST` | Standard agent chat response |
| `/api/ai/chat/stream` | `POST` | Streaming chat response over SSE |
| `/api/search` | `POST` | Agentic property search |
| `/api/properties` | `GET` | Paginated property listing |
| `/api/users/register` | `POST` | User registration |
| `/api/users/login` | `POST` | User login |
| `/api/users/<id>/ai/conversations` | `GET/POST` | Load or create conversations |
| `/api/users/<id>/favorites` | `GET/POST` | Favorite management |
| `/api/recommendations` | `GET` | Personalized recommendations |

## Testing

Run the test suite with:

```bash
pytest tests -q -p no:cacheprovider
```

The project includes regression tests for:

- agent routing and handoff
- query parsing and follow-up context
- streaming SSE behavior
- Hugging Face and `llama.cpp` fallback behavior
- live listing parameter normalization
- favorites, chat memory, and CRUD flows

## Troubleshooting

### Live listings are not appearing

Check:

- `RENTCAST_API_KEY` is set
- your network allows outbound requests
- your prompt includes a city, ZIP code, or neighborhood

### Chat keeps falling back to local reasoning

Check:

- `HF_DISABLED` is not forcing hosted inference off
- `LLAMA_CPP_ENABLED` is set correctly
- your `llama.cpp` server is reachable at `LLAMA_CPP_URL`
- your Hugging Face token and chosen model/provider are valid if using hosted inference

### Docker cannot reach `llama.cpp`

When running in Docker, the app uses:

```text
http://host.docker.internal:8080/v1/chat/completions
```

Make sure your local `llama.cpp` server is running and bound so the host can serve that endpoint.

### Old UI state or stale scripts

If the UI looks outdated after changes:

- restart the Flask app or container
- hard refresh the browser with `Ctrl+F5`

## Production Notes

This project currently uses SQLite for simplicity. For a more serious deployment, consider:

- PostgreSQL instead of SQLite
- reverse proxying with Nginx or Caddy
- structured application logging
- background workers for heavier ingestion or analytics flows
- secret management outside `.env`

## License

Add the license that matches your intended usage before public release.
