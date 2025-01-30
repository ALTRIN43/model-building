**1. Pandas: Data Manipulation and Analysis**

Pandas is an essential Python library for data manipulation and analysis. It provides powerful data structures such as DataFrames and Series, making it easy to load, clean, and analyze structured data.

Key Features:

Reading and writing data from CSV, Excel, SQL, and JSON.

Handling missing values efficiently.

Data filtering, grouping, and transformation using functions like groupby(), apply(), and merge().

Time series analysis capabilities.

Example:

import pandas as pd
# Load data from a CSV file
df = pd.read_csv("data.csv")
# Display first five rows
print(df.head())

**2. NumPy: Numerical Computation**

NumPy (Numerical Python) is a foundational package for numerical computations in Python. It provides high-performance multi-dimensional arrays and tools for matrix operations.

Key Features:

Supports large, multi-dimensional arrays and matrices.

Provides mathematical functions such as linear algebra, Fourier transforms, and random number generation.

Optimized performance using C-based implementations.

Example:

import numpy as np
# Creating a NumPy array
arr = np.array([1, 2, 3, 4, 5])
# Perform mathematical operations
print(arr * 2)  # Multiply each element by 2

**3. Scikit-Learn: Machine Learning**

Scikit-Learn (sklearn) is a popular library for machine learning and data mining. It provides simple and efficient tools for building and evaluating models.

Key Features:

Supports classification, regression, clustering, and dimensionality reduction.

Built-in functions for data preprocessing like feature scaling and encoding.

Provides model selection tools such as cross-validation.

Integrated with other scientific computing libraries like NumPy and Pandas.

Example:

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Splitting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

**4. Seaborn: Statistical Data Visualization**

Seaborn is built on top of Matplotlib and provides beautiful, informative statistical graphics.

Key Features:

Simplifies complex visualization tasks.

Supports themes for aesthetically pleasing plots.

Provides functions for visualizing categorical, numerical, and relational data.

Example:

import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("tips")

# Create a scatter plot
sns.scatterplot(x="total_bill", y="tip", hue="sex", data=df)
plt.show()

**5. Matplotlib: Basic Plotting**

Matplotlib is a fundamental library for plotting graphs and visualizing data in Python.

Key Features:

Supports various plot types, including line, bar, scatter, and histograms.

Highly customizable with options for labels, legends, colors, and styles.

Used as the backend for Seaborn and Pandas visualizations.

Example:

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label='Sine Wave')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Simple Line Plot")
plt.legend()
plt.show()

Conclusion

These libraries provide a powerful ecosystem for working with data in Python. Whether you are performing data analysis, building machine learning models, or visualizing insights, these tools will help streamline your workflow
