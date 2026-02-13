import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# STEP 1 — Load Dataset
# ===============================
data = pd.read_csv("data/students.csv", encoding="utf-8")
data.columns = data.columns.str.strip()

# ===============================
# STEP 2 — Features & Target
# ===============================
X = data[["attendance", "internal_marks", "study_hours", "assignments"]]
y = data["performance"].map({"Pass": 1, "Fail": 0})

# ===============================
# STEP 3 — Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 4 — Train Model
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# STEP 5 — Evaluate Model
# ===============================
print("\nModel Accuracy:", model.score(X_test, y_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test)))

# ===============================
# STEP 6 — USER INPUT (IMPORTANT CHANGE)
# ===============================
print("\nEnter Student Details:")

attendance = float(input("Attendance (%): "))
internal_marks = float(input("Internal Marks: "))
study_hours = float(input("Study Hours per Day: "))
assignments = float(input("Assignment Score: "))

new_student = pd.DataFrame(
    [[attendance, internal_marks, study_hours, assignments]],
    columns=["attendance", "internal_marks", "study_hours", "assignments"]
)

prediction = model.predict(new_student)[0]
result = "PASS" if prediction == 1 else "FAIL"

print("\nPrediction:", result)

# ===============================
# STEP 7 — PROMPT: Explanation
# ===============================
def explanation_prompt(student, prediction):
    a, m, h, asg = student.iloc[0]

    explanation = f"""
AI Explanation:
The model analyzed attendance ({a}%), internal marks ({m}),
study hours ({h}), and assignment performance ({asg}).

"""

    if prediction == "PASS":
        explanation += (
            "The student is likely to PASS due to consistent academic behavior "
            "and sufficient engagement across learning activities."
        )
    else:
        explanation += (
            "The student is likely to FAIL due to insufficient academic engagement "
            "or weak performance indicators."
        )

    return explanation

# ===============================
# STEP 8 — PROMPT: Recommendations
# ===============================
def recommendation_prompt(student):
    a, m, h, asg = student.iloc[0]

    rec = "\nAI Recommendations:\n"

    if a < 75:
        rec += "- Improve attendance to at least 75%.\n"
    if m < 35:
        rec += "- Strengthen internal exam preparation.\n"
    if h < 3:
        rec += "- Increase daily study hours.\n"
    if asg < 70:
        rec += "- Improve assignment submission quality.\n"

    if rec == "\nAI Recommendations:\n":
        rec += "- Maintain current academic consistency.\n"

    return rec

# ===============================
# STEP 9 — DISPLAY PROMPT OUTPUT
# ===============================
print(explanation_prompt(new_student, result))
print(recommendation_prompt(new_student))



