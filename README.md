# Credit Card Fraud Detection

Credit Card Fraud Detection is a behavioral fraud detection platform that combines sequence modeling, feature engineering, and explainable AI to identify anomalous financial transactions.

The system models user transaction history as a temporal pattern and uses that context to estimate risk, surface decisions, and support model training workflows.

## Stack

- Backend: FastAPI on Python 3.14
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
├── docker-compose.yml
└── package.json
```

## Run Locally

From the repository root:

```bash
docker compose up --build
```

You can also use the npm shortcuts:

```bash
npm start
```

Available scripts:

- `npm start`: starts both containers with a fresh build
- `npm run start:detached`: starts the stack in the background
- `npm run stop`: stops the stack
- `npm run restart`: stops the stack and rebuilds it
- `npm run logs`: streams container logs
- `npm run ps`: shows the current container status

## Endpoints

- Frontend: http://localhost:3000
- Backend root: http://localhost:8000
- Health check: http://localhost:8000/api/health
- API health: http://localhost:8000/api/v1/health
- Prediction: `POST http://localhost:8000/api/v1/predict`
- Training jobs:
  - `POST /api/v1/training/jobs`
  - `GET /api/v1/training/jobs`
  - `GET /api/v1/training/jobs/{job_id}`
  - `POST /api/v1/training/jobs/{job_id}/cancel`

## Notes

- The frontend proxies `/api/*` requests to the backend container through Nginx.
- CORS is configured for `http://localhost:3000` by default.
- Training job execution relies on the files under `backend/training/` and the paths configured in `backend/app/core/config.py`.
