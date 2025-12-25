## Synthetic dataset 
## Contains features like brand condition specs prices and target variable "returned"
import numpy as np          
import pandas as pd         
from pathlib import Path    


rng = np.random.default_rng(42) # random number generator object


OUT = Path(__file__).parent / "smartphone_returns.csv"## to save the dataset in the same folder as the file in

## We define possible values for each features

brands = np.array(["Apple", "Samsung", "Google", "OnePlus", "Xiaomi"])
conditions = np.array(["New", "OpenBox", "Refurbished"])
ram_gb_choices = np.array([6, 8, 12, 16])
storage_gb_choices = np.array([64, 128, 256, 512])
carriers = np.array(["Unlocked", "Verizon", "AT&T", "T-Mobile"])

N = 5000  ## Total number of rows 

## rng.choice(array,size,p=probabilities)
## Probabilities of each feature the categories will occur 
## To make it a market trend we have do this 

brand = rng.choice(brands, N, p=[0.36, 0.32, 0.14, 0.10, 0.08])
condition = rng.choice(conditions, N, p=[0.70, 0.20, 0.10])
ram_gb = rng.choice(ram_gb_choices, N, p=[0.20, 0.45, 0.25, 0.10])
storage_gb = rng.choice(storage_gb_choices, N, p=[0.10, 0.40, 0.35, 0.15])
carrier = rng.choice(carriers, N, p=[0.55, 0.15, 0.15, 0.15])

## normal means normal distribution with mean and standard deviation
## and also clip between higher and lower value
price = rng.normal(850, 220, N).clip(200, 2000)
usd_discount = rng.normal(80, 60, N).clip(0, 600)
user_rating = rng.normal(4.4, 0.5, N).clip(1.0, 5.0)
num_support_tickets_30d = rng.poisson(0.6, N).clip(0, 10)
days_since_purchase = rng.integers(1, 120, N)
expedited_shipping = rng.choice([0, 1], N, p=[0.7, 0.3])

## Base Probability
base = 0.08 ## Every phone has this amount of return probability

## Brand returning probability
brand_bias = (
    np.where(brand == "Google", 0.03, 0.0) +
    np.where(brand == "Xiaomi", 0.02, 0.0)
)
## Condition Bias
condition_bias = (
    np.where(condition == "Refurbished", 0.07, 0.0) +
    np.where(condition == "OpenBox", 0.03, 0.0)
)
## Price Bias cheap phone -> Higher returning chances
price_bias = np.clip((1200 - price) / 3000, -0.05, 0.06)

rating_bias = np.clip((4.0 - user_rating) / 5.0, 0.0, 0.10)

support_bias = np.clip(num_support_tickets_30d * 0.02, 0.0, 0.2)

shipping_bias = np.where(expedited_shipping == 1, 0.01, 0.0)


propensity = (
    base +
    brand_bias +
    condition_bias +
    price_bias +
    rating_bias +
    support_bias +
    shipping_bias
)


propensity = np.clip(propensity, 0.01, 0.60)


returned = rng.binomial(1, propensity)


frame = pd.DataFrame({
    "brand": brand,
    "condition": condition,
    "ram_gb": ram_gb,
    "storage_gb": storage_gb,
    "carrier": carrier,
    "price": price.round(2),
    "usd_discount": usd_discount.round(2),
    "user_rating": user_rating.round(2),
    "num_support_tickets_30d": num_support_tickets_30d,
    "days_since_purchase": days_since_purchase,
    "expedited_shipping": expedited_shipping,
    "returned": returned,
})


frame.to_csv(OUT, index=False)


print(
    f"Wrote {OUT} with shape {frame.shape} | "
    f"return rate={frame['returned'].mean():.2%}"
)