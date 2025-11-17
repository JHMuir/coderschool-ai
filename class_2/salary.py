import pandas
import numpy
import matplotlib
import sklearn

# Pandas: Stands for Panel Data. Organizes our dataset into a chart-like object. VERY popular with data scientists when working with .csv
# We can use Pandas DataFrames very similarly to dictionaries
# SciKit Learn (SkLearn): A ML library similar to Keras. Main difference being, Sklearn has more options for data preprocessing, and comes with certain models out of box
# Keras provides more freedom when constructing a model yourself. 
# ===== Step 1: Data Pre-Processing ===== 

# Load the dataset
df = pandas.read_csv('Salary_dataset.csv')

# Display basic information about the dataset
print(df.head())

print("\nDataset shape (rows, columns):", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Separate features (X) and target (y)
# X = independent variable (what we use to make predictions)
# y = dependent variable (what we want to predict)
x = df[['YearsExperience']]  # Features - Data we are using to predict other variables. Keep as DataFrame for scikit-learn
y = df['Salary']  # What we want to predict

print(f"\nFeature (X) shape: {x.shape}")
print(f"Target (y) shape: {y.shape}")

# Split the data into training and testing sets
# Training set: Used to teach the model the pattern
# Testing set: Used to evaluate how well the model works on new data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(x_train)} samples")
print(f"Testing set size: {len(x_test)} samples")

# Plotting our data
matplotlib.pyplot.figure(figsize=(10, 6))
matplotlib.pyplot.scatter(x_train, y_train, color='blue', alpha=0.6, label='Training Data')
matplotlib.pyplot.scatter(x_test, y_test, color='red', alpha=0.6, label='Testing Data')
matplotlib.pyplot.xlabel('Years of Experience', fontsize=12)
matplotlib.pyplot.ylabel('Salary ($)', fontsize=12)
matplotlib.pyplot.title('Salary vs Years of Experience', fontsize=14, fontweight='bold')
matplotlib.pyplot.legend()
matplotlib.pyplot.grid(True, alpha=0.3)
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.show()

# ===== Step 2: Training ===== 

# Create a Linear Regression model
# Linear Regression finds the best-fit line through the data points
# The line equation is: y = mx + b
#   where: m = slope (how much salary increases per year of experience)
#          b = intercept (predicted salary at 0 years experience)
model = sklearn.linear_model.LinearRegression()

# Train the model on the training data
# This is where the model "learns" the relationship between x and y
model.fit(x_train, y_train)

print(f"\nModel Parameters:")
print(f"  Slope (coefficient): ${model.coef_[0]:.2f} per year")
print(f"  Intercept: ${model.intercept_:.2f}")
print(f"\n  Interpretation: For each additional year of experience,")
print(f"  salary increases by approximately ${model.coef_[0]:.2f}")

# ===== Step 3: Inferencing =====

# Make predictions on the test set
y_pred = model.predict(x_test)

# Display some example predictions
for i in range(min(10, len(x_test))):
    years = x_test.iloc[i, 0]
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    print(f"{years:<20.1f} ${actual:<19,.0f} ${predicted:<19,.0f}")

# Calculate evaluation metrics
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
rmse = numpy.sqrt(mse)
mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
r2 = sklearn.metrics.r2_score(y_test, y_pred)

# Create a range of experience values for plotting the line
X_range = numpy.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_range = model.predict(X_range)

matplotlib.pyplot.figure(figsize=(12, 6))

# Plot 1: Regression line with data points
matplotlib.pyplot.subplot(1, 2, 1)
matplotlib.pyplot.scatter(x_train, y_train, color='blue', alpha=0.5, label='Training Data')
matplotlib.pyplot.scatter(x_test, y_test, color='red', alpha=0.5, label='Testing Data')
matplotlib.pyplot.plot(X_range, y_range, color='green', linewidth=2, label='Regression Line')
matplotlib.pyplot.xlabel('Years of Experience', fontsize=11)
matplotlib.pyplot.ylabel('Salary ($)', fontsize=11)
matplotlib.pyplot.title('Linear Regression Model', fontsize=12, fontweight='bold')
matplotlib.pyplot.legend()
matplotlib.pyplot.grid(True, alpha=0.3)
matplotlib.pyplot.show()

# Plot 2: Actual vs Predicted values
matplotlib.pyplot.subplot(1, 2, 2)
matplotlib.pyplot.scatter(y_test, y_pred, color='purple', alpha=0.6)
matplotlib.pyplot.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
matplotlib.pyplot.xlabel('Actual Salary ($)', fontsize=11)
matplotlib.pyplot.ylabel('Predicted Salary ($)', fontsize=11)
matplotlib.pyplot.title('Actual vs Predicted Salaries', fontsize=12, fontweight='bold')
matplotlib.pyplot.legend()
matplotlib.pyplot.grid(True, alpha=0.3)
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.show()

