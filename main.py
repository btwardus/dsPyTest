import dspy
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import requests

from src.module import ProductResolutionModule
from src.data import load_train_data
from dspy.teleprompt import BootstrapFewShot

# =====================
# Configuration Constants
# =====================
# Model Provider Selection: Set to one of "ollama", "openai", or "openrouter"
MODEL_PROVIDER = "openrouter"  # Options: "ollama", "openai", "openrouter"

# Ollama Configuration
OLLAMA_MODEL_NAME = 'qwen3:14b'  # Name of the Ollama model to use

# OpenAI Configuration  
OPENAI_MODEL_NAME = 'gpt-4.1'  # Name of the openai model to use

# OpenRouter Configuration
OPENROUTER_MODEL_NAME = 'openrouter/mistralai/codestral-2508'  # Name of the OpenRouter model to use
SYSTEM_PROMPT = (
    "You are an entity resolution assistant. For each product pair, output only 'Yes' or 'No' as the label, and nothing else as the first line. "
    "Then, provide a short, step-by-step explanation of your reasoning as the next field. "
    "If you are uncertain, prefer 'Yes'."
)  # System prompt for the language model
TRAIN_FILE_PATH = 'data/train_products_tailored.csv'  # Path to the training CSV file
VALIDATION_FILE_PATH = 'data/val_products.csv'  # Path to the validation CSV file
VALIDATION_SAMPLE_SIZE = 10  # Number of validation examples to sample (start small)
MAX_WORKERS = 2  # Number of threads for parallel evaluation
MAX_BOOTSTRAPPED_DEMOS = 1  # Number of bootstrapped demos for BootstrapFewShot
# =====================

def extract_label(output):
    output = output.strip().splitlines()[0]  # Take the first line, strip whitespace
    if output.lower() in {"yes", "no"}:
        return output.capitalize()
    match = re.search(r'\b(Yes|No)\b', output, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "Unknown"

def format_product(row, prefix):
    name = row[f'name_{prefix}'] if f'name_{prefix}' in row else ''
    description = row[f'description_{prefix}'] if f'description_{prefix}' in row else ''
    manufacturer = row[f'manufacturer_{prefix}'] if f'manufacturer_{prefix}' in row else ''
    price = row[f'price_{prefix}'] if f'price_{prefix}' in row else ''
    if prefix == 'abt':
        return f"Name: {name} Description: {description} Price: {price}"
    else:
        return f"Name: {name} Description: {description} Manufacturer: {manufacturer} Price: {price}"

# Simple metric for use with BootstrapFewShotWithRandomSearch
def simple_metric(example, prediction, trace=None):
    return getattr(example, "label", None) == getattr(prediction, "label", None)

def evaluate_teleprompter(teleprompter, teleprompter_name, train_subset, val_df, max_workers=MAX_WORKERS):
    print(f"\n{'='*30}\nEvaluating {teleprompter_name}\n{'='*30}")
    product_resolution_module = ProductResolutionModule()
    compiled_program = teleprompter.compile(product_resolution_module, trainset=train_subset)
    print(f"Compilation complete for {teleprompter_name}.")

    def predict_label(idx, row):
        product1 = format_product(row, 'abt')
        product2 = format_product(row, 'buy')
        result = compiled_program(product1=product1, product2=product2)
        raw_label = str(result.label)
        explanation = result.explanation
        confidence = getattr(result, 'confidence', None)
        pred = extract_label(raw_label)
        actual = str(row['label']).strip().capitalize()
        return idx, actual, pred, raw_label, explanation, confidence, product1, product2

    y_true, y_pred = [], []
    disagreements = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(predict_label, idx, row): idx for idx, row in val_df.iterrows()}
        for i, future in enumerate(as_completed(futures), 1):
            idx, actual, pred, raw_label, explanation, confidence, product1, product2 = future.result()
            y_true.append(actual)
            y_pred.append(pred)
            print(f"Example {i}: True={actual} | Pred={pred} | Confidence={confidence if confidence is not None else '[No confidence]'}")
            if pred != actual:
                disagreements.append((i, actual, pred, raw_label, explanation, confidence, product1, product2, val_df.loc[idx].to_dict()))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["Yes", "No"]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=["Yes", "No"]))

    if disagreements:
        print("\n--- Disagreement Details ---")
        for idx, actual, pred, raw_label, explanation, confidence, product1, product2, row_dict in disagreements:
            print(f"\nExample {idx}:")
            print(f"  True Label: {actual}")
            print(f"  Predicted Label: {pred}")
            print(f"  Raw Output: {raw_label}")
            print(f"  Model's Explanation: {explanation if explanation else '[No explanation provided]'}")
            print(f"  Confidence: {confidence if confidence is not None else '[No confidence]'}")
            print(f"  Product 1 (formatted): {product1}")
            print(f"  Product 2 (formatted): {product2}")
        print("--- End of Disagreements ---")
    else:
        print("\nNo disagreements found.")

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Configure the language model based on provider selection
    if MODEL_PROVIDER == "ollama":
        # Check if Ollama is available
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise ValueError("Ollama is not running. Please start Ollama first.")
        except Exception as e:
            raise ValueError(f"Ollama connection failed: {e}")
        
        lm = dspy.OllamaLocal(
            model=OLLAMA_MODEL_NAME,
            base_url="http://localhost:11434",
            max_tokens=2048,
            timeout_s=420
        )
        print(f"Using Ollama model: {OLLAMA_MODEL_NAME}")
        
    elif MODEL_PROVIDER == "openai":
        # Check OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        lm = dspy.OpenAI(
            model=OPENAI_MODEL_NAME, 
            api_key=openai_api_key, 
            system_prompt=SYSTEM_PROMPT
        )
        print(f"Using OpenAI model: {OPENAI_MODEL_NAME}")
        
    elif MODEL_PROVIDER == "openrouter":
        # Check OpenRouter API key
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        lm = dspy.LM(
            model=OPENROUTER_MODEL_NAME,
            api_base="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        print(f"Using OpenRouter model: {OPENROUTER_MODEL_NAME}")
        
    else:
        raise ValueError(f"Invalid MODEL_PROVIDER: {MODEL_PROVIDER}. Must be one of: ollama, openai, openrouter")

    # Configure DSPy with the selected language model
    dspy.configure(lm=lm)

    # Use all training data for compilation
    train_data = load_train_data(TRAIN_FILE_PATH)
    if not isinstance(train_data, list) or len(train_data) == 0:
        print("Error: train_data is empty or not a list. Please check your training data.")
        sys.exit(1)
    train_subset = train_data
    print(f"Using {len(train_subset)} training examples.")

    # Load validation data from CSV
    if not os.path.exists(VALIDATION_FILE_PATH):
        print(f"Error: Validation file {VALIDATION_FILE_PATH} does not exist.")
        sys.exit(1)
    val_df = pd.read_csv(VALIDATION_FILE_PATH)
    if len(val_df) == 0:
        print(f"Error: Validation file {VALIDATION_FILE_PATH} is empty.")
        sys.exit(1)
    val_df = val_df.reset_index(drop=True)
    # Sample validation data if more than VALIDATION_SAMPLE_SIZE
    if len(val_df) > VALIDATION_SAMPLE_SIZE:
        val_df = val_df.sample(n=VALIDATION_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Using {len(val_df)} validation examples.")

    # Metric function in main.py
    def get_metric(example, prediction, trace=None):
        # Use the same label extraction logic as in the evaluation loop
        predicted_label = extract_label(prediction.label)
        # Ensure comparison is case-insensitive
        return example.label.strip().lower() == predicted_label.strip().lower()

    # Try different teleprompters
    teleprompters = [
        (BootstrapFewShot(metric=get_metric, max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS), "BootstrapFewShot")
    ]

    for teleprompter, name in teleprompters:
        evaluate_teleprompter(teleprompter, name, train_subset, val_df)

if __name__ == "__main__":
    main() 