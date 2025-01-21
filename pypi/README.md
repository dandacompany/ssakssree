# ssakssree

A collection of statistics-based business analysis libraries by Danda Company, a professional analytics firm

## Installation

```bash
pip install ssakssree
```

## Key Features

- Mixed Model-based Conjoint Analysis
- Price scaling support
- Baseline level specification
- Interaction analysis between attributes
- Price range recommendation based on target rating

## Usage

```python
from ssakssree.recommender import MixedConjointPriceRecommender
import pandas as pd

# Config
config = {
    'respondent_col': 'RespondentID',
    'rating_col': 'Rating',
    'price_col': 'Price',
    'categorical_cols': ['Brand', 'SolutionQuality', 'Difficulty', 'Design'],
    'interaction_cols': ['Brand', 'Design'],
    'price_scale_factor': 1000,
    'baseline_levels': {
        'Brand': 'Local',
        'SolutionQuality': 'Basic',
        'Difficulty': 'Easy',
        'Design': '단색'
    }
}

# Load data & fit model
df = pd.read_csv("your_data.csv")
recommender = MixedConjointPriceRecommender(config)
recommender.fit_model(df)

# Find price range for a specific attribute combination
attributes = {
    'Brand': 'National',
    'SolutionQuality': 'Advanced',
    'Difficulty': 'Easy',
    'Design': '단색'
}

# Find price range for a specific attribute combination
price_range = recommender.find_price_range(
    attributes,
    price_min=5000,
    price_max=30000,
    threshold=4.5
)
print(f"Recommended price range: {price_range}")
```
