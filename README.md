# AI Engineer for Data Scientists: Customer Purchase Prediction

A PyTorch neural network that predicts whether an online shopper will make a purchase, based on how they browse the website.

## Business problem
RetailTech Solutions, an international e-commerce platform, wants to know which customers are most likely to buy. Accurate predictions help the marketing team target the right customers and plan expansion into new markets.

## Data
Each row is one customer session:

| Column | Description |
|---|---|
| customer_id | Unique customer identifier |
| time_spent | Minutes spent on the website in the session |
| pages_viewed | Number of pages viewed in the session |
| basket_value | Value of items in the basket |
| device_type | Mobile, Desktop or Tablet |
| customer_type | New or Returning |
| purchase | Target: 1 if the customer bought, 0 if not |

The original data files belong to DataCamp and are not included in this repository.

## Approach

### 1. Data cleaning
- Converted every column to the correct data type
- Filled missing values by column rules: median for time spent, mean for pages viewed, zero for basket value
- Standardised text categories (for example "mobile" and " MOBILE " became "Mobile")
- Replaced missing device type with "Unknown" and missing customer type with "New"

### 2. Feature preparation
- Scaled time spent, pages viewed and basket value to a 0 to 1 range with Min-Max scaling
- One-hot encoded device type and customer type into columns such as device_type_Desktop and customer_type_Returning

### 3. Neural network
| Layer | Details |
|---|---|
| Input | One unit per feature |
| Hidden | 8 units, ReLU activation |
| Output | 1 unit, Sigmoid activation (probability of purchase) |

- Loss function: binary cross-entropy
- Optimiser: Adam, learning rate 0.01
- Training: 500 epochs, fixed random seed for reproducible results
- Prediction: a probability of 0.5 or higher counts as a purchase

## Results
- Training accuracy: [your value, for example 0.81]
- Pass requirement: accuracy above 0.75 on a held-out test set

## Repository structure

AI-Engineer-for-Data-Scientists/
├── notebook.ipynb # data cleaning, feature preparation, model training and predictions
└── README.md

## Tools
Python, pandas, NumPy, PyTorch

## Certification
Completed as the practical exam for the DataCamp AI Engineer for Data Scientists certification.
