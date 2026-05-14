# Credit Card Fraud Detection

Credit Card Fraud Detection is a behavioral fraud detection platform that combines sequence modeling, feature engineering, and explainable AI to identify anomalous financial transactions.

The system models user transaction history as a temporal pattern and uses that context to estimate risk, surface decisions, and support model training workflows.

## Stack

- Backend: FastAPI on Python 3.14
- Database: PostgreSQL 16
- Frontend: HTML, CSS, and JavaScript served by Nginx
- Orchestration: Docker Compose

## Dataset

The project uses the following reference dataset:

- Kaggle: https://www.kaggle.com/datasets/kartik2112/fraud-detection?utm_source=chatgpt.com

## Project Layout

```text
credit-card-fraud-detection/
├── backend/
│   ├── app/
│   │   ├── controllers/
│   │   ├── core/
│   │   ├── domain/
│   │   ├── features/
│   │   ├── ml/
│   │   ├── repositories/
│   │   ├── schemas/
│   │   └── services/
│   ├── training/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.database.yml
├── docker-compose.backend.yml
├── docker-compose.frontend.yml
└── package.json
```

## Run Locally

From the repository root:

```bash
npm run startAll
```

You can also use the npm shortcuts:

```bash
npm start
```

Available scripts:

- `npm start`: starts PostgreSQL, backend, and frontend with a fresh build
- `npm run start:detached`: starts the full stack in the background
- `npm run startAll`: explicit alias for `npm start`
- `npm run startAll:detached`: explicit alias for `npm run start:detached`
- `npm run startBack`: starts PostgreSQL and backend
- `npm run startDatabase`: starts only PostgreSQL in the background
- `npm run stop`: stops the stack
- `npm run restart`: stops the stack and rebuilds it
- `npm run restartAll`: explicit alias for `npm run restart`
- `npm run logs`: streams container logs
- `npm run ps`: shows the current container status

## Endpoints

- Frontend: http://localhost:3000
- Backend root: http://localhost:8000
- PostgreSQL: localhost:5432
- Health check: http://localhost:8000/api/health
- API health: http://localhost:8000/api/v1/health
- Prediction: `POST http://localhost:8000/api/v1/predict`
- Training jobs:
  - `POST /api/v1/training/jobs`
  - `GET /api/v1/training/jobs`
  - `GET /api/v1/training/jobs/{job_id}`
  - `POST /api/v1/training/jobs/{job_id}/cancel`

## AI Analysis LLM Provider

AI analysis uses the local LLM configuration by default:

- `LOCAL_LLM_BASE_URL`
- `LOCAL_LLM_MODEL`
- `LOCAL_LLM_API_KEY`

To route AI analysis through OpenRouter, set both values below. If either one is missing or empty, the backend falls back to the local LLM.

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`

`OPENROUTER_BASE_URL` defaults to `https://openrouter.ai/api/v1/chat/completions`.

### LangSmith / LangChain tracing

To inspect LangChain executions in the LangSmith dashboard while keeping the frontend AI Analysis flow, configure:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_ENDPOINT`

The AI Analysis screen shows the tracing status and an **Open LangSmith** link. Traces are emitted for LangChain `ChatOpenAI` compatible providers. If `LOCAL_LLM_BASE_URL` ends with `/api/v1/chat`, the app preserves the current LM Studio direct API flow used by the frontend, but that direct HTTP path is not traced by LangSmith. To trace local LM Studio calls, use an OpenAI-compatible base URL such as `http://host.docker.internal:1234/v1` when available.

## LangGraph Studio

The project includes a LangGraph configuration at `langgraph.json` with a `fraud_analysis` graph that reuses the backend AI Analysis service.

Run it with:

```bash
npm run langgraph:serve
```

The npm script pins the LangGraph CLI runtime to `python3.12` through `UV_PYTHON`, avoiding Python 3.13 compatibility issues in the local development server. It also rewrites the LangGraph process database URL to `localhost`, because the Docker-only hostname `postgres` is not resolvable from the host. For local LM Studio, set `LANGGRAPH_LOCAL_LLM_BASE_URL` when the host process needs a different URL than the backend container; WSL usually works with `http://host.docker.internal:1234/v1`. The server defaults to `http://localhost:2024`. The graph input accepts:

```json
{
  "question": "Which imported transactions deserve analyst attention?",
  "filters": {
    "source": "analyzed",
    "limit": 25
  }
}
```

## Notes

- The frontend proxies `/api/*` requests to the backend container through Nginx.
- CORS is configured for `http://localhost:3000` by default.
- Training job execution relies on the files under `backend/training/` and the paths configured in `backend/app/core/config.py`.
