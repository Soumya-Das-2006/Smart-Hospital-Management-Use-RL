---
title: Smart Hospital Resource Allocator
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🏥 Smart Hospital Resource Allocator — OpenEnv RL Environment

An RL environment where an LLM agent manages ICU beds, doctors, nurses, and ambulances
across an 8-hour emergency department shift.
Built for the **Meta PyTorch OpenEnv Hackathon 2026**.

---

## 🗂 Project Structure

```
smart_hospital_web/
├── inference.py                 ← ✅ Hackathon submission script
├── Dockerfile                   ← ✅ Container build (port 7860)
├── openenv.yaml                 ← ✅ OpenEnv config
├── pyproject.toml               ← ✅ UV-compatible dependencies
├── requirements.txt             ← Python dependencies
├── run.py                       ← Flask entry point
│
├── core/
│   ├── env.py                   ← HospitalEnv (Gymnasium, 7 actions, 480 steps)
│   ├── config.py                ← All constants (beds, rewards, PPO/DQN cfg)
│   └── agents.py                ← Heuristic + random agent
│
└── app/
    ├── __init__.py              ← Flask app factory
    ├── routes/
    │   ├── api.py               ← /health /reset /step /state + /api/* rich endpoints
    │   └── views.py             ← HTML page routes
    ├── services/
    │   └── hospital_service.py  ← Thread-safe session manager
    ├── templates/               ← Dashboard, admin, docs pages
    └── static/                  ← CSS + JS
```

---

## 🔌 OpenEnv Standard Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Container health check (required by OpenEnv spec) |
| `POST` | `/reset`  | Start new episode → returns `session_id` |
| `POST` | `/step`   | Take action `{"session_id":"...","action":0}` |
| `GET`  | `/state`  | Get current state without stepping |

## 🌐 Web Interface

| URL | Description |
|-----|-------------|
| `/` | Landing page |
| `/dashboard` | Live interactive RL dashboard |
| `/admin` | Admin panel + downloads |
| `/docs` | Full API reference |

---

## ⚡ Quick Setup

```bash
cd smart_hospital_web
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
# → http://localhost:7860
```

---

## 🚀 Inference Script

```bash
export HF_TOKEN=your_hf_token_here
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

---

## 🎮 Action Space

| ID | Name | Description |
|----|------|-------------|
| 0 | `admit_to_icu` | Admit critical patient to ICU (needs doctor + nurse) |
| 1 | `admit_to_general` | Admit patient to general ward |
| 2 | `dispatch_ambulance` | Send ambulance; 60% chance brings critical patient |
| 3 | `escalate_to_icu` | Upgrade deteriorating general patient to ICU |
| 4 | `discharge_patient` | Discharge recovered patient, free a bed |
| 5 | `call_extra_staff` | +2 doctors +3 nurses for 60 timesteps |
| 6 | `hold_and_monitor` | Wait and monitor this timestep |

---

## 📊 Reward Structure

| Signal | Value |
|--------|-------|
| Patient survived | +100 |
| Correct triage | +30 |
| Wrong triage | -25 |
| Patient death | -200 |
| Patient deterioration | -60 |
| Idle ICU bed | -1.5/step |
| Overcrowded general ward | -25 |
| High throughput bonus | +8 |

**Grader signal**: `survival_rate` from `info` dict — normalized `0.0–1.0`.
Success threshold: `survival_rate >= 0.50`.

---

## 🏭 Docker Deploy

```bash
docker build -t hospital-rl .
docker run -p 7860:7860 -e HF_TOKEN=your_token hospital-rl
```

# Smart-Hospital-Management-Use-RL
