# Personalized News Recommendation on MIND

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-NRMS-ee4c2c) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Course project for **CSE 482 (Big Data Analysis)** comparing two approaches to personalized news recommendation on the [Microsoft News Dataset (MIND-small)](https://msnews.github.io/): a classical **TF-IDF content-based filter** and a neural **NRMS** (Neural News Recommendation with Multi-Head Self-Attention) model trained on BERT-tokenized headlines.

## Highlights

- Built user profiles from click history and ranked candidate impressions by cosine similarity of TF-IDF article vectors (51K articles × 157K features).
- Implemented NRMS in PyTorch with a Hugging Face tokenizer; trained on ~36.5K impression batches and evaluated on the full 73K-impression dev set.
- Evaluated both models with standard ranking metrics: AUC, MRR, nDCG@5 and nDCG@10.
- Results — TF-IDF: AUC 0.591, MRR 0.338, nDCG@10 0.373; NRMS: AUC 0.612, MRR 0.312, nDCG@10 0.360.

## Contents

- [`notebooks/01_mind_data_exploration.ipynb`](notebooks/01_mind_data_exploration.ipynb) — loading and inspecting `news.tsv` / `behaviors.tsv`
- [`notebooks/02_tfidf_content_filtering.ipynb`](notebooks/02_tfidf_content_filtering.ipynb) — TF-IDF content-based recommender and evaluation
- [`notebooks/03_nrms_neural_recommender.ipynb`](notebooks/03_nrms_neural_recommender.ipynb) — NRMS model, training loop, checkpointing and evaluation

## Repository Structure

```text
mind-news-recommendation/
├── notebooks/
│   ├── 01_mind_data_exploration.ipynb
│   ├── 02_tfidf_content_filtering.ipynb
│   └── 03_nrms_neural_recommender.ipynb
├── LICENSE
├── README.md
└── requirements.txt
```

## Tech Stack

Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas, NumPy

## Getting Started

```bash
git clone https://github.com/siddakmarwaha/mind-news-recommendation.git
cd mind-news-recommendation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Data

Download **MINDsmall_train** and **MINDsmall_dev** from https://msnews.github.io/ and unzip them into `notebooks/` (the notebooks read `./MINDsmall_train/` and `./MINDsmall_dev/`). Trained model checkpoints (`*.pth`, ~150 MB) are not committed.

Note: the NRMS notebook imports helper code from a local `model_utils` module that was not part of the saved files; the notebook outputs show the full training and evaluation run.

## Author

**Siddak Marwaha**

## License

Code in this repository is released under the [MIT License](LICENSE).
