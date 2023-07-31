import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRFRegressor


from sklearn.datasets import make_regression

# Generate a sample regression dataset
X, y = make_regression(n_samples=1000000, n_features=10, random_state=42)

# Print the shape of the dataset
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Step 1: Install necessary libraries (if not already installed)

# Step 2: Set GPU parameters
params = {
    "tree_method": "gpu_hist",
    "gpu_id": 0  # Specify GPU ID if you have multiple GPUs
}

# Step 3: Prepare your data
# Load and preprocess your dataset using pandas or other libraries

# Step 4: Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and fit the XGBRFRegressor model
model = XGBRFRegressor(n_estimators=100, max_depth=6, **params)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
