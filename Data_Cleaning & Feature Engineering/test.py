import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Sample DataFrame
df = pd.DataFrame({
    'area': [1000, 1500, None, 1200],
    'bedrooms': [2, 3, 3, None],
    'bathrooms': [1, None, 2, 1]
})

# Function to fill NA with mode, keeping column names
def fill_mode(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        mode_val = df_copy[col].mode()[0]
        df_copy[col] = df_copy[col].fillna(mode_val)
    return df_copy

# Pipeline
pipeline = Pipeline([
    ('mode_imputer', FunctionTransformer(fill_mode, validate=False))
])

# ✅ Correct method call
result = pipeline.fit_transform(df)

print(result)
