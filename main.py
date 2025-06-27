import dspy
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.module import ProductResolutionModule
from src.data import train_data
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot, BootstrapFewShotWithRandomSearch

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

def evaluate_teleprompter(teleprompter, teleprompter_name, train_subset, val_df, max_workers=5):
    print(f"\n{'='*30}\nEvaluating {teleprompter_name}\n{'='*30}")
    product_resolution_module = ProductResolutionModule()
    compiled_program = teleprompter.compile(product_resolution_module, trainset=train_subset)
    print(f"Compilation complete for {teleprompter_name}.")

    def predict_label(row):
        product1 = format_product(row, 'abt')
        product2 = format_product(row, 'buy')
        result = compiled_program(product1=product1, product2=product2)
        raw_label = str(result.label)
        pred = extract_label(raw_label)
        actual = str(row['label']).strip().capitalize()
        return actual, pred, raw_label, product1, product2

    y_true, y_pred = [], []
    disagreements = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(predict_label, row) for _, row in val_df.iterrows()]
        for idx, (row_idx, future) in enumerate(zip(val_df.index, as_completed(futures)), 1):
            actual, pred, raw_label, product1, product2 = future.result()
            y_true.append(actual)
            y_pred.append(pred)
            print(f"Example {idx}: True={actual} | Pred={pred}")
            if pred != actual:
                disagreements.append((idx, actual, pred, raw_label, product1, product2, val_df.loc[row_idx].to_dict()))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["Yes", "No"]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=["Yes", "No"]))

    if disagreements:
        print("\n--- Disagreement Details ---")
        for idx, actual, pred, raw_label, product1, product2, row_dict in disagreements:
            print(f"\nExample {idx}:")
            print(f"  True Label: {actual}")
            print(f"  Predicted Label: {pred}")
            print(f"  Raw Output: {raw_label}")
            print(f"  Product 1 (formatted): {product1}")
            print(f"  Product 2 (formatted): {product2}")
        print("--- End of Disagreements ---")
    else:
        print("\nNo disagreements found.")

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Configure the language model
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    turbo = dspy.OpenAI(model='gpt-4.1', api_key=openai_api_key, system_prompt="You are an entity resolution assistant. For each product pair, output only 'Yes' or 'No' as the label, and nothing else. The label must be the first line of your response. If you are uncertain, prefer 'Yes'.")
    dspy.settings.configure(lm=turbo)

    # Use all training data for compilation
    train_subset = train_data
    print(f"Using {len(train_subset)} training examples.")

    # Load validation data from CSV
    val_df = pd.read_csv('data/val_products.csv').sample(n=200, random_state=42).reset_index(drop=True)

    # Try different teleprompters
    teleprompters = [
        (BootstrapFewShot(metric=None, max_bootstrapped_demos=2), "BootstrapFewShot")
    ]

    for teleprompter, name in teleprompters:
        evaluate_teleprompter(teleprompter, name, train_subset, val_df)

if __name__ == "__main__":
    main() 