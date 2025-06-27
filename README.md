# Entity Resolution with DSPy: Abt-Buy Product Matching

## Overview
This project demonstrates entity resolution (record linkage) for product data using [DSPy](https://github.com/stanfordnlp/dspy) and large language models (LLMs). It matches product records from two e-commerce sources (Abt and Buy.com) and predicts whether they refer to the same real-world product.

## Dataset
- **Source:** [Abt-Buy Entity Resolution Benchmark](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution#:~:text=Used%20in-,Abt%2DBuy,-E%2Dcommerce) ([Leipzig Database Group](https://dbs.uni-leipzig.de/))
- **Files:**
  - `Abt.csv`, `Buy.csv`: Product catalogs from two online shops
  - `abt_buy_perfectMapping.csv`: Ground-truth matching pairs
- **How to Download:**
  1. Visit the [Leipzig ER Benchmark page](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution#:~:text=Used%20in-,Abt%2DBuy,-E%2Dcommerce)
  2. Download the "Abt-Buy" dataset zip and extract the three CSVs into `data/raw/`

## Project Structure
```
.
├── data/
│   ├── raw/                # Original Abt-Buy data
│   ├── train_products.csv  # Training pairs (generated)
│   ├── val_products.csv    # Validation pairs (generated)
│   └── validation_results_bootstrapfewshot_2025-06-26.md  # Example results
├── src/
│   ├── data.py             # Data loading and formatting
│   ├── module.py           # DSPy module definition
│   └── signature.py        # DSPy signature for product matching
├── create_train_val_split.py # Script to generate train/val splits
├── main.py                 # Main experiment script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup
1. **Install Python 3.9+** and [pip](https://pip.pypa.io/en/stable/).
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your OpenAI API key:**
   - Create a `.env` file in the project root with:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Data Preparation
1. **Download and extract the Abt-Buy dataset** as described above.
2. **Generate train/validation splits:**
   ```bash
   python create_train_val_split.py
   ```
   This will create `data/train_products.csv` and `data/val_products.csv`.

## Running the Experiment
Run the main script to evaluate entity resolution using DSPy and OpenAI GPT-4.1:
```bash
python main.py
```
- The script will:
  - Compile a DSPy program using few-shot learning (BootstrapFewShot)
  - Evaluate on a sample of the validation set
  - Print confusion matrix, classification report, and disagreement details

## Customization
- **Prompt/Model:** Edit `main.py` to change the system prompt, model, or number of few-shot demos.
- **Validation Size:** Adjust the sample size or use the full validation set as needed.
- **Teleprompters:** Try other DSPy teleprompters (see `main.py` for examples).

## Advanced Teleprompter Optimization

BootstrapFewShot is a great starting point. For potentially better performance, you can explore `dspy.teleprompt.BootstrapFewShotWithRandomSearch`. This optimizer not only finds the best examples but also searches for the best instructions and prompt formats. It requires more computational effort but can yield superior results.

## Results Overview

### Example Results: `gpt-4.1` 439 validation cases

```
              precision    recall  f1-score   support

         Yes       1.00      0.98      0.99       197
          No       0.99      1.00      0.99       242

    accuracy                           0.99       439
   macro avg       0.99      0.99      0.99       439
weighted avg       0.99      0.99      0.99       439
```

This demonstrates high accuracy and balanced performance for both classes.

### Example Results: `gemma:2b` (Ollama, 50 validation cases)

```
             precision    recall  f1-score   support

         Yes       0.57      0.18      0.28        22
          No       0.00      0.00      0.00        28

   micro avg       0.57      0.08      0.14        50
   macro avg       0.29      0.09      0.14        50
weighted avg       0.25      0.08      0.12        50
```

## References
- [Abt-Buy Dataset and Benchmark](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution#:~:text=Used%20in-,Abt%2DBuy,-E%2Dcommerce)
- [DSPy: Modular LLM Pipelines](https://github.com/stanfordnlp/dspy)

---
*For research and educational use. Cite the Leipzig Database Group and VLDB2010 paper if publishing results.* 