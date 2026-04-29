# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("Everything working ✅")

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("iris.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Remove unwanted columns like 'id' (IMPORTANT FIX)
df = df.drop(columns=["id"], errors="ignore")

print("Columns after cleaning:", df.columns)

# -------------------------------
# Step 2: Prepare Data
# -------------------------------
# Last column is target
target_column = df.columns[-1]

X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert labels to numbers
le = LabelEncoder()
y = le.fit_transform(y)

# -------------------------------
# Step 3: Split Data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: Train Model
# -------------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 6: Evaluation
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# -------------------------------
# Step 7: Test Custom Input
# -------------------------------
print("\nEnter values for prediction:")
sepal_length = float(input("Sepal Length: "))
sepal_width = float(input("Sepal Width: "))
petal_length = float(input("Petal Length: "))
petal_width = float(input("Petal Width: "))

sample = [[sepal_length, sepal_width, petal_length, petal_width]]

prediction = model.predict(sample)
print("Predicted Flower:", le.inverse_transform(prediction))

# -------------------------------
# Step 8: Visualization
# -------------------------------
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Iris Dataset Visualization")
plt.show()