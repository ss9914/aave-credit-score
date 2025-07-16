import json
import pandas as pd

with open('data/transactions.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(df.head())
print(df.columns)
