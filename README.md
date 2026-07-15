<h1 align="center">FE-SyncMap</h1>

<p align="center">
  <b>Free-Energy-Inspired Online Unsupervised Chunk Discovery from Streaming Sequential Data</b>
</p>

<p align="center">
  🧠 Online Unsupervised Learning · 🔄 Streaming Sequential Data · 🧩 Chunk Discovery · ⚡ Self-Organization
</p>

## ✨ Overview

This repository provides the implementation and experimental resources for **FE-SyncMap**, an online unsupervised method for discovering latent chunk structures from streaming sequential data.

FE-SyncMap introduces local information signals based on **surprisal** and **uncertainty** into the SyncMap self-organizing framework. It supports continual structure discovery without labels, replay buffers, or global optimization.

<p align="center">
  <img src="results/dynamic/fixed5_120---probabilistic5_120/fe_syncmap_static_dynamic_dbscan_20avg.png" width="760" alt="FE-SyncMap dynamic structure-switching result"/>
</p>

## 💡 Key innovations

- **Free-energy-inspired online self-organization**  
  Introduces local information signals into SyncMap while preserving fully online, unsupervised learning without labels, replay buffers, or a global optimization objective.

- **Dual information-guided candidate selection**  
  Uses **surprisal** to prioritize informative positive relations and **uncertainty** to guide negative candidate selection, allowing the model to focus on structurally meaningful interactions.

- **Information-adaptive geometric updates**  
  Modulates symmetric attraction–repulsion updates according to local information content, improving the quality and stability of the learned chunk structure.

- **Robust continual chunk discovery**  
  Supports structural reorganization under non-stationary streams and performs especially well on probabilistic, community-based, and imbalanced chunk structures.

## 📦 What’s inside

- 🧩 **FE-SyncMap implementation**
  - Core model files in `models/`
- 🔁 **Sequential stream generation**
  - Graph loading and random-walk utilities in `datasets/`
- 🧪 **Experiment scripts**
  - Training and evaluation files in `train/`
- 🗂️ **CGCP benchmark data**
  - Fixed, probabilistic, mixed, imbalanced, and unequal-size chunk graphs in `data/`
- 🌐 **Real-world network data**
  - Dolphin social network and Zachary’s Karate Club in `data/real/`
- 📊 **Experimental results**
  - Benchmark results, dynamic-switching curves, CSV files, and visualizations in `results/`

## 🗂️ Repository structure

```text
FE-Syncmap/
├── data/
│   ├── *.dot                         # CGCP benchmark graphs
│   └── real/
│       ├── dolphins.gml              # Dolphin social network
│       └── revised_kc.gml            # Zachary's Karate Club
├── datasets/
│   └── GraphProcessor.py             # graph and sequence processing
├── models/
│   └── FE_SyncMap.py                 # FE-SyncMap model
├── train/
│   └── train.py                      # experiment entry
├── utils/                             # metrics and visualization utilities
├── results/
│   ├── dynamic Visualization/        # online embedding animations
│   ├── dynamic/                      # structure-switching results
│   └── results.xlsx                  # benchmark results
└── README.md
```

## 📬 Contact

For questions about the project, please contact **Jinyi Zhang** at `zhangjinyi@sylu.edu.cn`.
