# PathMind — RL Traffic Signal Optimizer

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=flat&logo=c%2B%2B&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat&logo=fastapi&logoColor=white)
![pybind11](https://img.shields.io/badge/pybind11-3.x-4B8BBE?style=flat&logo=python&logoColor=white)

This project tackles the problem of road congestion using a reinforcement learning system. An RL agent learns to control traffic signals across 10 intersections to minimize the number of cars waiting at any given time.

The simulation environment is written in C++ for speed — training runs 100k+ steps, so response time matters. The C++ environment is bridged to Python using pybind11, which compiles it into a `.pyd` file that Python imports like any normal module.

The agent is a policy network built in PyTorch — it takes a 30-element observation (signal states, car volumes, occupancy across all intersections) and outputs a probability distribution over 10 actions. PPO (Proximal Policy Optimization) trains this network by running episodes, computing discounted returns, and updating weights using a clipped surrogate loss to prevent unstable updates.

Results are displayed on a live FastAPI + WebSocket dashboard showing per-intersection car volumes, signal states, and a real-time reward chart.

---

## Results

| Agent | Total Reward |
|---|---|
| Random baseline | -32,232 |
| Trained PPO agent | **-3,080** |

**~10x improvement** over random action selection.

---

## Architecture

```
┌─────────────────────────┐
│  C++ TrafficEnvironment │  (TrafficEnvironment.cpp)
│  10 intersections       │  step(), reset()
└────────────┬────────────┘
             │  compiled via
             ▼
┌────────────────────────┐
│       pybind11         │  (bindings.cpp)
└────────────┬───────────┘
             │  imports as
             ▼
┌────────────────────────┐
│   traffic_env.pyd      │  Python-callable C++ module
└────────────┬───────────┘
             │  used by
             ▼
┌────────────────────────┐
│  Python PPO  train.py  │  collect episodes, compute returns,
│                        │  clipped surrogate loss, Adam update
└────────────┬───────────┘
             │  trains
             ▼
┌────────────────────────┐
│  PolicyNetwork         │  Linear(30→64)→ReLU→Linear(64→10)→Softmax
│  agent.py              │  saved as trained_policy.pth
└────────────┬───────────┘
             │  loaded by
             ▼
┌────────────────────────┐
│  FastAPI  api/main.py  │  runs inference every 500 ms
│                        │  background thread + lifespan
└────────────┬───────────┘
             │  streams JSON via
             ▼
┌────────────────────────┐
│     WebSocket /ws      │  step, action, car_volume,
│                        │  signal_states, reward
└────────────┬───────────┘
             │  rendered by
             ▼
┌────────────────────────┐
│  Dashboard  index.html │  10 intersection cards,
│                        │  live reward chart, dark theme
└────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation environment | C++17 |
| Python–C++ bridge | pybind11 |
| RL algorithm | PPO (custom PyTorch implementation) |
| Neural network | PyTorch 2.x |
| Backend / API | FastAPI + Uvicorn |
| Real-time streaming | WebSockets |
| Frontend | Vanilla HTML/JS (no framework) |

---

## How to Run

### 1. Build the C++ environment

```bash
cd pathmind
cmake -S . -B build
cmake --build build --config Release
```

This compiles `traffic_env.pyd` into `build/Release/`.

### 2. Install Python dependencies

```bash
pip install torch fastapi "uvicorn[standard]"
```

### 3. Train the PPO agent

```bash
cd pathmind
python python/train.py
```

Saves `trained_policy.pth` on completion.

### 4. Launch the dashboard

```bash
cd pathmind
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.
