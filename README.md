# Credit Card Fraud Detection

Credit Card Fraud Detection is a behavioral fraud detection platform that combines sequence modeling, feature engineering, and explainable AI to identify anomalous financial transactions.

The system models user transaction history as a temporal pattern and uses that context to estimate risk, surface decisions, and support model training workflows.

<img width="1904" height="1057" alt="image" src="https://github.com/user-attachments/assets/e90b0f1c-23fb-4604-b967-2cc807e3b98f" />

https://github.com/user-attachments/assets/4e86fd05-c0a5-4233-96ce-ea45075c08af

## Stack

- Backend: FastAPI on Python 3.14 in Docker
- Database: PostgreSQL 16
- Frontend: HTML, CSS, and JavaScript served by Nginx
- ML training: scikit-learn baseline and PyTorch GRU/LSTM sequence models
- AI analysis: local LM Studio/OpenAI-compatible LLM or OpenRouter
- Observability: optional LangSmith tracing and LangGraph Studio
- Orchestration: Docker Compose with npm scripts

## Dataset

The project uses the Kaggle synthetic credit card fraud dataset:

- Kaggle: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- Expected training file: `backend/training/data/fraudTrain.csv`
- Expected test/import file: `backend/training/data/fraudTest.csv`

Create the dataset directory before downloading:

```bash
mkdir -p backend/training/data
```

### Manual Download

1. Open https://www.kaggle.com/datasets/kartik2112/fraud-detection.
2. Download the dataset ZIP.
3. Extract the files.
4. Move the CSV files into:

```text
backend/training/data/fraudTrain.csv
backend/training/data/fraudTest.csv
```

### Kaggle CLI Download

Configure your Kaggle API token first. Kaggle normally expects `~/.kaggle/kaggle.json`.

Run from the repository root:

```bash
mkdir -p backend/training/data
kaggle datasets download -d kartik2112/fraud-detection -p backend/training/data --unzip
```

Confirm the files:

```bash
ls -lh backend/training/data/fraudTrain.csv backend/training/data/fraudTest.csv
```

## Project Layout

```text
credit-card-fraud-detection/
├── backend/
│   ├── app/
│   │   ├── controllers/        # FastAPI route handlers
│   │   ├── core/               # Environment-driven settings
│   │   ├── domain/             # Transaction, risk, and decision domain objects
│   │   ├── features/           # Runtime feature building
│   │   ├── langgraph/          # LangGraph fraud analysis graph
│   │   ├── llm/                # Local LLM/OpenRouter clients
│   │   ├── ml/                 # Model loading and inference
│   │   ├── repositories/       # PostgreSQL persistence
│   │   ├── schemas/            # Pydantic request/response models
│   │   └── services/           # Application services
│   ├── migrations/             # SQL migrations
│   ├── scripts/                # Database, cleanup, and LangGraph helpers
│   ├── tests/                  # Unit and integration tests
│   ├── training/
│   │   ├── data/               # Place Kaggle CSV files here
│   │   ├── artifacts/          # Generated models, metrics, logs, registry
│   │   ├── specs/              # Feature specifications
│   │   ├── train_baseline.py
│   │   └── train_sequence.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── public/                 # Static HTML entrypoint
│   ├── src/                    # App shell, modules, services, styles
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.database.yml
├── docker-compose.backend.yml
├── docker-compose.frontend.yml
├── langgraph.json
└── package.json
```

## Prerequisites

- Docker and Docker Compose
- Node.js with npm
- Kaggle account for the dataset download
- Optional: Python 3.12 on the host for `npm run langgraph:serve`
- Optional: LM Studio or another OpenAI-compatible local LLM server
- Optional: OpenRouter API key
- Optional: LangSmith API key

## Environment

Create a local `.env` file:

```bash
cp .env.example .env
```

Important defaults:

```env
POSTGRES_DB=fraud_detection
POSTGRES_USER=fraud_user
POSTGRES_PASSWORD=fraud_password
POSTGRES_PORT=5432
DATABASE_URL=postgresql://fraud_user:fraud_password@postgres:5432/fraud_detection
LOCAL_LLM_BASE_URL=http://host.docker.internal:1234/v1
LOCAL_LLM_MODEL=openai/gpt-oss-20b
LOCAL_LLM_API_KEY=lm-studio
LANGGRAPH_LOCAL_LLM_BASE_URL=http://host.docker.internal:1234/v1
AI_ANALYSIS_MAX_TRANSACTIONS=200
```

Use `DATABASE_URL` with host `postgres` for Docker containers. The LangGraph helper rewrites database access to `localhost` because LangGraph runs on the host.

## Run Locally

From the repository root:

```bash
npm start
```

This builds and starts:

- PostgreSQL
- FastAPI backend
- Nginx frontend

Application URLs:

- Frontend: http://localhost:3000
- Backend root: http://localhost:8000
- API health: http://localhost:8000/api/v1/health
- PostgreSQL: localhost:5432

## Recommended First Run

1. Create `.env`.

```bash
cp .env.example .env
```

2. Download and place the Kaggle CSV files.

```bash
mkdir -p backend/training/data
kaggle datasets download -d kartik2112/fraud-detection -p backend/training/data --unzip
```

3. Start the database.

```bash
npm run startDatabase
```

4. Apply migrations.

```bash
npm run migrate:database
```

5. Start the full stack.

```bash
npm start
```

6. Open the frontend.

```text
http://localhost:3000
```

7. Train a model from the Model Monitoring screen.

8. Import `fraudTest.csv` from the Transactions screen and select which model should score the imported transactions.

Available scripts:

- `npm start`: starts PostgreSQL, backend, and frontend with a fresh build
- `npm run start:detached`: starts the full stack in the background
- `npm run startBack`: starts PostgreSQL and backend
- `npm run startDatabase`: starts only PostgreSQL in the background
- `npm run startFront`: recreates only the frontend container
- `npm run stop`: stops the stack
- `npm run restart`: stops the stack and rebuilds it
- `npm run clean:database`: truncates transactions and AI analysis history
- `npm run migrate:database`: applies SQL migrations
- `npm run migrate:database:down`: rolls SQL migrations back
- `npm run langgraph:serve`: starts the LangGraph development server
- `npm run logs`: streams container logs
- `npm run ps`: shows the current container status

## Database Migrations And Cleanup

Start the database first:

```bash
npm run startDatabase
```

Apply migrations:

```bash
npm run migrate:database
```

Rollback migrations:

```bash
npm run migrate:database:down
```

Clean all persisted transaction and AI-analysis data:

```bash
npm run clean:database
```

The cleanup command truncates `transactions` and `ai_analysis_history`, including imported transactions.

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
  - `DELETE /api/v1/training/jobs/{job_id}`
  - `GET /api/v1/training/datasets`
- Transactions:
  - `GET /api/v1/transactions`
  - `POST /api/v1/transactions/import/fraud-test`
  - `GET /api/v1/transactions/import-jobs/{job_id}`
- Dashboard:
  - `GET /api/v1/dashboard/overview`
- AI Analysis:
  - `POST /api/v1/ai-analysis/query`
  - `GET /api/v1/ai-analysis/observability`
  - `GET /api/v1/ai-analysis/history`
  - `DELETE /api/v1/ai-analysis/history/{analysis_id}`

## Training

The training UI is available in the frontend under Model Monitoring.

Training jobs read datasets from:

```text
backend/training/data/
```

Generated artifacts are written to:

```text
backend/training/artifacts/jobs/<job_id>/
```

Each job persists metadata in:

```text
backend/training/artifacts/jobs_registry.json
```

### Train Through The API

Baseline model:

```bash
curl -X POST http://localhost:8000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "baseline",
    "dataset_path": "backend/training/data/fraudTrain.csv",
    "run_name": "baseline-fraud-train"
  }'
```

Sequence model:

```bash
curl -X POST http://localhost:8000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sequence",
    "dataset_path": "backend/training/data/fraudTrain.csv",
    "run_name": "sequence-gru-fraud-train",
    "sequence_config": {
      "backbone": "gru",
      "seq_len": 20,
      "stride": 10,
      "batch_size": 128,
      "epochs": 5
    }
  }'
```

List training jobs:

```bash
curl http://localhost:8000/api/v1/training/jobs
```

List datasets visible to the backend:

```bash
curl http://localhost:8000/api/v1/training/datasets
```

## Import And Analyze `fraudTest.csv`

The import workflow expects:

```text
backend/training/data/fraudTest.csv
```

Use the Transactions screen import control. The import will:

- Read `fraudTest.csv`
- Analyze each row with the selected trained model, or the default model when no training job is selected
- Save imported and analyzed rows in PostgreSQL
- Make imported transactions visible in Dashboard and Transactions
- Make those transactions available to AI Analysis

API example:

```bash
curl -X POST http://localhost:8000/api/v1/transactions/import/fraud-test \
  -H "Content-Type: application/json" \
  -d '{
    "batch_size": 1000,
    "training_job_id": "<optional-trained-model-job-id>"
  }'
```

Check import status:

```bash
curl http://localhost:8000/api/v1/transactions/import-jobs/<job_id>
```

## Prediction API

Manual transaction scoring:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 91.25,
    "transaction_datetime": "2020-12-31T23:59:34",
    "merchant": "fraud_Dare-Marvin",
    "category": "entertainment",
    "gender": "F",
    "state": "CA",
    "job": "Engineer",
    "city_population": 120000,
    "customer_latitude": 34.05,
    "customer_longitude": -118.24,
    "merchant_latitude": 34.06,
    "merchant_longitude": -118.25,
    "transactions_last_hour": 1,
    "transactions_last_24h": 3,
    "average_amount_24h": 82.10
  }'
```

Optional field:

```json
{
  "training_job_id": "<trained-model-job-id>"
}
```

## Dashboard

Dashboard reads from PostgreSQL and includes:

- Transaction count
- Known fraud rate
- Review/reject rate
- Average risk score
- Fraud trend
- Decision breakdown
- Category risk
- Hourly volume
- Latest transactions
- Live alerts

Imported `fraudTest.csv` transactions appear after the import job finishes.

## AI Analysis LLM Provider

AI analysis uses the local LLM configuration by default:

- `LOCAL_LLM_BASE_URL`
- `LOCAL_LLM_MODEL`
- `LOCAL_LLM_API_KEY`
- `AI_ANALYSIS_MAX_TRANSACTIONS`

`AI_ANALYSIS_MAX_TRANSACTIONS` is the backend safety cap for the number of transactions sent to the model. It defaults to `200`, which matches the maximum accepted by the AI Analysis request schema and the frontend transaction limit control.

To route AI analysis through OpenRouter, set both values below. If either one is missing or empty, the backend falls back to the local LLM.

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`

`OPENROUTER_BASE_URL` defaults to `https://openrouter.ai/api/v1/chat/completions`.

Example `.env` configuration:

```env
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=<your-openrouter-api-key>
```

Restart the backend after changing these values:

```bash
npm run restart
```

The OpenRouter client sends a chat-completions request with bearer-token authentication, JSON messages, the selected model, low temperature, and `response_format={"type":"json_object"}` so the AI Analysis service can parse a structured response.

For LM Studio, enable the local server and expose an OpenAI-compatible endpoint. In WSL/Docker flows, `host.docker.internal` is usually the correct host:

```env
LOCAL_LLM_BASE_URL=http://host.docker.internal:1234/v1
LOCAL_LLM_MODEL=openai/gpt-oss-20b
LOCAL_LLM_API_KEY=lm-studio
```

### LangSmith / LangChain tracing

To inspect LangChain executions in the LangSmith dashboard while keeping the frontend AI Analysis flow, configure:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_ENDPOINT`

The AI Analysis screen shows the tracing status and an **Open LangSmith** link. Traces are emitted for LangChain `ChatOpenAI` compatible providers. If `LOCAL_LLM_BASE_URL` ends with `/api/v1/chat`, the app preserves the current LM Studio direct API flow used by the frontend, but that direct HTTP path is not traced by LangSmith. To trace local LM Studio calls, use an OpenAI-compatible base URL such as `http://host.docker.internal:1234/v1` when available.

## LangGraph Studio

The project includes a LangGraph configuration at `langgraph.json` with a `fraud_analysis` graph that reuses the backend AI Analysis service.

<img width="1658" height="891" alt="Screenshot 2026-05-14 164235" src="https://github.com/user-attachments/assets/01cfb858-de4e-43a6-9bec-d2a9b665e605" />

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
    "limit": 50
  }
}
```

## Notes

- The frontend proxies `/api/*` requests to the backend container through Nginx.
- CORS is configured for `http://localhost:3000` by default.
- Training job execution relies on the files under `backend/training/` and the paths configured in `backend/app/core/config.py`.

## Testing

Run backend unit tests from the repository root:

```bash
pytest backend/tests/unit -q
```

Run selected AI-analysis tests:

```bash
pytest backend/tests/unit/test_langchain_client.py backend/tests/unit/test_ai_analysis_service.py -q
```

## Troubleshooting

### `fraudTrain.csv was not found`

Place the file at:

```text
backend/training/data/fraudTrain.csv
```

### `fraudTest.csv was not found`

Place the file at:

```text
backend/training/data/fraudTest.csv
```

### LangGraph cannot resolve host `postgres`

Use the npm wrapper:

```bash
npm run langgraph:serve
```

Do not call `npx @langchain/langgraph-cli@latest dev` directly unless you also provide host-compatible environment variables.

### Local LLM is unavailable

Confirm that your LLM server is running:

```bash
curl http://host.docker.internal:1234/v1/models
```

If this fails, update `LOCAL_LLM_BASE_URL` or `LANGGRAPH_LOCAL_LLM_BASE_URL` in `.env`.

### Database is empty

Run migrations and import data:

```bash
npm run startDatabase
npm run migrate:database
```

Then import `fraudTest.csv` from the Transactions screen or through the import API.
