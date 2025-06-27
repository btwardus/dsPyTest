import pandas as pd
from sklearn.model_selection import train_test_split
import random

def clean_text(text):
    if pd.isna(text):
        return ''
    return str(text).replace('\n', ' ').replace('\r', ' ').replace('"', "'").strip()

# Load data with encoding handling
abt = pd.read_csv('data/raw/Abt.csv', encoding='ISO-8859-1')
buy = pd.read_csv('data/raw/Buy.csv', encoding='ISO-8859-1')
mapping = pd.read_csv('data/raw/abt_buy_perfectMapping.csv', encoding='ISO-8859-1')

# Merge to get positive pairs
positive_pairs = mapping.merge(abt, left_on='idAbt', right_on='id') \
                        .merge(buy, left_on='idBuy', right_on='id', suffixes=('_abt', '_buy'))

positive_examples = [
    {
        'idAbt': row['idAbt'],
        'name_abt': clean_text(row['name_abt']),
        'description_abt': clean_text(row['description_abt']),
        'price_abt': clean_text(row['price_abt']),
        'idBuy': row['idBuy'],
        'name_buy': clean_text(row['name_buy']),
        'description_buy': clean_text(row['description_buy']),
        'manufacturer_buy': clean_text(row['manufacturer']),
        'price_buy': clean_text(row['price_buy']),
        'label': 'Yes'
    }
    for _, row in positive_pairs.iterrows()
]

# Generate negative pairs (not in mapping)
abt_ids = set(abt['id'])
buy_ids = set(buy['id'])
positive_set = set(zip(mapping['idAbt'], mapping['idBuy']))
negatives = []
while len(negatives) < len(positive_examples):
    a = random.choice(list(abt_ids))
    b = random.choice(list(buy_ids))
    if (a, b) not in positive_set:
        row_abt = abt[abt['id'] == a].iloc[0]
        row_buy = buy[buy['id'] == b].iloc[0]
        negatives.append({
            'idAbt': a,
            'name_abt': clean_text(row_abt['name']),
            'description_abt': clean_text(row_abt['description']),
            'price_abt': clean_text(row_abt['price']),
            'idBuy': b,
            'name_buy': clean_text(row_buy['name']),
            'description_buy': clean_text(row_buy['description']),
            'manufacturer_buy': clean_text(row_buy['manufacturer']),
            'price_buy': clean_text(row_buy['price']),
            'label': 'No'
        })

# Combine and split
data = positive_examples + negatives
random.shuffle(data)
train, val = train_test_split(data, test_size=0.2, random_state=42)

# Save as CSV
pd.DataFrame(train).to_csv('data/train_products.csv', index=False)
pd.DataFrame(val).to_csv('data/val_products.csv', index=False)

print('Train and validation CSVs created: data/train_products.csv, data/val_products.csv') 