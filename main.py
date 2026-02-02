# Fix matplotlib freeze issue (IMPORTANT – keep this at top)
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# ===============================
# STEP 1 — Load Dataset
# ===============================
data = pd.read_csv("data/students.csv")

# Clean column names
data.columns = data.columns.str.strip()

# ===============================
# STEP 2 — Features & Target
# ===============================
X = data[["attendance", "internal_marks", "study_hours", "assignments"]]
y = data["performance"].map({"Pass": 1, "Fail": 0})  # Encode target

# ===============================
# STEP 3 — Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 4 — Train Model
# ===============================
model = LogisticRegression()
model.fit(X_train, y_train)

# ===============================
# STEP 5 — Model Accuracy
# ===============================
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# ===============================
# STEP 6 — Predict New Student
# ===============================
new_student = pd.DataFrame(
    [[85, 40, 3, 80]],
    columns=["attendance", "internal_marks", "study_hours", "assignments"]
)

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction for new student: PASS")
else:
    print("Prediction for new student: FAIL")

# ===============================
# STEP 7 — VISUALIZATIONS
# ===============================

# GRAPH 1 — Pass vs Fail Count
sns.countplot(x=data["performance"])
plt.title("Pass vs Fail Distribution")
plt.xlabel("Performance")
plt.ylabel("Number of Students")
plt.show()

# GRAPH 2 — Attendance vs Performance
sns.boxplot(x=data["performance"], y=data["attendance"])
plt.title("Attendance Impact on Performance")
plt.xlabel("Performance")
plt.ylabel("Attendance")
plt.show()

# GRAPH 3 — Internal Marks vs Performance
sns.boxplot(x=data["performance"], y=data["internal_marks"])
plt.title("Internal Marks Impact on Performance")
plt.xlabel("Performance")
plt.ylabel("Internal Marks")
plt.show()

# ===============================
# STEP 8 — Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, model.predict(X_test))
print("Confusion Matrix:\n", cm)



