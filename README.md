🚀 Tensor Risk Engine

Tensor Risk Engine is a behavioral fraud detection system that leverages sequence modeling and explainable AI to identify anomalous financial transactions.

🧠 Overview

Traditional fraud detection systems rely on static rules or simple classification models.
This project introduces a sequence-based approach, where each user’s transaction history is modeled as a temporal pattern.

By using deep learning and tensor representations, the system learns normal behavior and detects deviations in real time.

## Stack Inicial

- Backend: FastAPI (Python 3.12)
- Frontend: HTML/CSS/JS servido por Nginx
- Orquestração: Docker Compose

## Estrutura

```text
btc-tensor-lab/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/routes/
│   │   ├── core/
│   │   ├── models/
│   │   ├── services/
│   │   ├── schemas/
│   │   ├── features/
│   │   ├── inference/
│   │   └── main.py
│   ├── training/
│   ├── notebooks/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── Dockerfile
│   └── nginx.conf
└── docker-compose.yml
```

## Como executar

Na raiz do projeto:

```bash
docker compose up --build
```

Ou usando scripts npm:

```bash
npm start
```

Scripts disponiveis:

- `npm start`: sobe os containers com build
- `npm run start:detached`: sobe em background
- `npm run stop`: derruba os containers
- `npm run restart`: reinicia com novo build
- `npm run logs`: acompanha logs em tempo real
- `npm run ps`: lista status dos servicos

## Endpoints e URLs

- Frontend: http://localhost:3000
- Backend root: http://localhost:8000
- Healthcheck: http://localhost:8000/api/health

## Observações

- O frontend usa proxy no Nginx para encaminhar `/api/*` para o container backend.
- O backend já está com CORS configurado para `http://localhost:3000`.
