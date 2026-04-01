# GPT-SEC: NLP-Driven SEC Filing Analysis for Market Sentiment Prediction

<p align="center">
  <strong>A hybrid deep learning pipeline that leverages NLP and time-series analysis to predict market sentiment from SEC filings.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--3.5-412991?logo=openai&logoColor=white" alt="OpenAI" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

---

## Overview

GPT-SEC is an end-to-end machine learning system that ingests SEC 10-Q and 10-K filings, generates natural language summaries via the OpenAI API, performs sentiment analysis using FinBERT, and combines textual features with historical price data through a hybrid Transformer + LSTM architecture to classify market reactions as **Buy**, **Sell**, or **Neutral**.

The core hypothesis: by processing the dense, standardized language of SEC filings before the broader market fully prices in their implications, a predictive model can capture actionable sentiment signals.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                              │
│                                                                     │
│  SEC EDGAR API ──▶ 10-Q / 10-K Filings ──▶ Section Text Extraction │
│  Polygon.io API ──▶ Minute-Level OHLCV Price Data                  │
└──────────────────────┬───────────────────────────┬──────────────────┘
                       │                           │
                       ▼                           ▼
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│      TEXT PIPELINE           │   │      TIME-SERIES PIPELINE        │
│                              │   │                                  │
│  GPT-3.5 Summarization      │   │  Feature Engineering             │
│  FinBERT Sentiment Analysis  │   │  (Close, Volatility, Timestamp) │
│  BERT Tokenization           │   │  Min-Max Normalization           │
│  Positional Encoding         │   │  Label Generation                │
└──────────────┬───────────────┘   └───────────────┬──────────────────┘
               │                                   │
               ▼                                   ▼
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│  TRANSFORMER ENCODER         │   │  LSTM NETWORK                    │
│  (6 layers, 8 heads, 128d)   │   │  (2 layers, 128 hidden units)   │
└──────────────┬───────────────┘   └───────────────┬──────────────────┘
               │                                   │
               └───────────┬───────────────────────┘
                           ▼
               ┌───────────────────────┐
               │   HYBRID CLASSIFIER   │
               │   (Concatenation +    │
               │    Dense Layer)       │
               │                       │
               │   ▶ Buy / Sell /      │
               │     Neutral           │
               └───────────────────────┘
```

---

## Key Features

**Automated Filing Ingestion** — Programmatic extraction of SEC 10-Q and 10-K filings via the SEC EDGAR API, with section-level text parsing across all standard filing categories.

**Intelligent Summarization** — Long-form filing sections are chunked to respect token limits, summarized independently via GPT-3.5 Turbo, and recombined into coherent section summaries.

**Financial Sentiment Analysis** — Section summaries are scored using [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone), a BERT model fine-tuned on financial text for three-class sentiment classification (Positive, Negative, Neutral).

**Volatility-Adjusted Labeling** — Buy/Sell/Neutral labels are generated dynamically based on price movement relative to rolling volatility, rather than fixed thresholds, adapting to each stock's behavior.

**Hybrid Deep Learning Model** — A custom architecture that fuses Transformer-encoded text representations with LSTM-encoded time-series features for joint classification.

**Broad Market Coverage** — Dataset spans 70+ S&P 500 constituents with approximately 10 quarterly filings each, covering 2020–2024, paired with minute-level intraday price data from Polygon.io.

---

## Dataset

| Component | Description | Scale |
|-----------|-------------|-------|
| **Text Dataset** | GPT-3.5 summaries of 10-Q filing sections, organized by ticker and filing date | ~700 filing summaries across 70+ tickers |
| **Stock Dataset** | Minute-level OHLCV data (open, high, low, close, volume, vwap) from Polygon.io | ~700 CSV files, each covering the trading day of the filing |
| **SEC Filings** | Raw section text from Tesla 10-K (Items 1–15) included as reference | 18 section files |

The text dataset is pre-generated and stored as `.txt` files, enabling model training without requiring API keys.

---

## Project Structure

```
gpt-sec/
├── gpt_sec_main.py        # Core pipeline: summarization, sentiment analysis, stock data retrieval
├── sec_query.py            # SEC EDGAR API wrapper (query, extract, section parsing)
├── text_handler.py         # Text chunking, tokenization, and OpenAI API interface
├── data_pipeline.py        # End-to-end data pipeline: text + stock dataset construction
├── models.py               # Model definitions (Transformer, LSTM, Hybrid, Positional Encoding)
├── construct_models.py     # Training loops, data loaders, evaluation, and label generation
├── tester.py               # Experimental training scripts and data loading utilities
├── train.py                # Standalone training entrypoints
├── text_dataset.json       # Pre-computed labeled text dataset (JSON)
├── Text Dataset/           # Summarized filing text organized by ticker/date
│   ├── AAPL/
│   │   ├── 2021-01-27.txt
│   │   └── ...
│   └── .../
├── Stock Dataset/          # Minute-level price data organized by ticker
│   ├── ABBV/
│   │   ├── ABBV0.csv
│   │   └── ...
│   └── .../
├── SEC_TSLA_0/             # Raw Tesla 10-K section text (reference data)
└── README.md
```

---

## Technical Stack

| Category | Technology |
|----------|------------|
| Deep Learning | PyTorch, Transformers (HuggingFace) |
| NLP Models | GPT-3.5 Turbo (summarization), FinBERT (sentiment), BERT Tokenizer |
| Data Sources | SEC EDGAR API (`sec-api`), Polygon.io REST API |
| Data Processing | pandas, NumPy, scikit-learn, NLTK |
| Tokenization | tiktoken (GPT token counting), BERT Tokenizer |
| Visualization | matplotlib |
| Serialization | pickle, JSON |

---

## Model Details

### Transformer Encoder (Text)
- Vocabulary: 30,522 tokens (BERT base uncased)
- Embedding dimension: 128
- Attention heads: 8
- Encoder layers: 6
- Feed-forward hidden dimension: 512
- Dropout: 0.1

### LSTM Network (Time-Series)
- Input features: 5 (hour, minute, second, close, volatility)
- Hidden units: 128
- Layers: 2
- Output: 3-class logits

### Hybrid Classifier
- Concatenates Transformer and LSTM output vectors
- Single fully-connected layer mapping to 3 classes (Buy, Sell, Neutral)
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr = 0.001)

---

## Getting Started

### Prerequisites

```bash
Python 3.10+
CUDA-compatible GPU (recommended)
```

### Installation

```bash
git clone https://github.com/Ayonator77/gpt-sec.git
cd gpt-sec
pip install torch transformers sec-api polygon-api-client tiktoken pandas scikit-learn nltk matplotlib
```

### API Keys

The following API keys are required for data collection (not needed if using the pre-built datasets):

| Service | Purpose | Environment Variable |
|---------|---------|---------------------|
| [SEC API](https://sec-api.io/) | Filing retrieval and section extraction | Set in `sec_query.py` |
| [OpenAI](https://platform.openai.com/) | GPT-3.5 Turbo summarization | Set in `text_handler.py` |
| [Polygon.io](https://polygon.io/) | Minute-level stock data | Set in `gpt_sec_main.py` |

### Training

```bash
# Train the hybrid model (requires preprocessed_stock_data.pkl)
python construct_models.py

# Or train individual components
python tester.py          # LSTM + Transformer experiments
python train.py           # Stock data pipeline
```

---

## Methodology

1. **Filing Retrieval** — Query SEC EDGAR for 10-Q filings by ticker, extracting each section (Part I Items 1–4, Part II Items 1–6) as raw text.

2. **Summarization** — Each section is chunked to fit within GPT-3.5 Turbo's context window (4K tokens). Sections exceeding the limit are split, summarized independently, and recombined through a second summarization pass.

3. **Sentiment Scoring** — FinBERT classifies each section summary as Positive, Negative, or Neutral with a confidence score.

4. **Price Data Alignment** — Minute-level OHLCV data is retrieved for the trading day of each filing. Rolling volatility is computed and used to generate adaptive Buy/Sell/Neutral labels.

5. **Feature Engineering** — Text summaries are tokenized using BERT's WordPiece tokenizer with padding and truncation. Time-series features include timestamp components and price/volatility metrics.

6. **Model Training** — The Transformer encoder processes tokenized text while the LSTM processes time-series features. Their outputs are concatenated and passed through a classification head. Training uses CrossEntropyLoss with the Adam optimizer.

7. **Evaluation** — Model performance is assessed via accuracy, loss curves, and ROC-AUC scores on held-out test data.

---

## Results & Status

The project is actively under development. Current model evaluation on validation data demonstrates the pipeline is functional end-to-end. Further work is focused on improving classification accuracy through architectural refinements, expanded training data, and hyperparameter optimization.

**Success criteria** (from project charter): A predictive model that outperforms the S&P 500 on back-tested and forward-tested data, or a definitive demonstration that no such model can achieve this.

---

## Future Work

- **Pre-trained Language Model Integration** — Replace the custom Transformer encoder with fine-tuned FinBERT or SEC-BERT for richer text representations.
- **Attention-Based Fusion** — Implement cross-attention between text and time-series streams instead of simple concatenation.
- **Expanded Label Window** — Experiment with multi-day price reaction windows (1-day, 5-day, 20-day) for label generation.
- **Backtesting Framework** — Build a portfolio simulation engine to evaluate model signals against S&P 500 returns.
- **Real-Time Pipeline** — Deploy as a streaming system that processes filings on publication for live signal generation.

---

## Author

**Ayodeji Odetola**
- GitHub: [@Ayonator77](https://github.com/Ayonator77)
- Twitter: [@ayodeji_odetola](https://twitter.com/ayodeji_odetola)

---

## License

This project is available under the [MIT License](LICENSE).

---

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Past performance of any model does not guarantee future results. Always conduct your own research before making investment decisions.
