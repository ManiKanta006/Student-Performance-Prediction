# Student Performance ML Pipeline Using Decision Tree

## Project Description

This college-level CSE AIML project builds a reproducible machine learning pipeline that predicts the observed `GradeClass` label for a student using a Decision Tree Classifier. The notebook is beginner-friendly and explains each stage from data loading through model saving.

`StudentID` is excluded because it is only an identifier. `GPA` is excluded because it is directly related to academic performance and could cause target leakage.

## Dataset Information

The project uses `Student_performance_data _.csv`, containing approximately 2,392 student records and these columns:

- `StudentID`
- `Age`
- `Gender`
- `Ethnicity`
- `ParentalEducation`
- `StudyTimeWeekly`
- `Absences`
- `Tutoring`
- `ParentalSupport`
- `Extracurricular`
- `Sports`
- `Music`
- `Volunteering`
- `GPA`
- `GradeClass`

The target classes are defined as `0 = A` (`GPA >= 3.5`), `1 = B` (`3.0 <= GPA < 3.5`), `2 = C` (`2.5 <= GPA < 3.0`), `3 = D` (`2.0 <= GPA < 2.5`), and `4 = F` (`GPA < 2.0`). `GPA` is documented for interpreting the target but is excluded from model inputs to prevent target leakage.

## Installation Instructions

Create and activate the virtual environment from PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name student-performance-venv --display-name "Python (Student Performance ML)"
```

## Required Libraries

The project requires Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter, and ipykernel. Python's built-in `pickle` module is used for model serialization, so no separate serialization package is required.

## How to Run the Notebook

1. Keep `Student_performance_data _.csv` and `Student_Performance_ML_Pipeline.ipynb` in the same folder.
2. Open the notebook in VS Code or Jupyter.
3. Select the `Python (Student Performance ML)` kernel.
4. Run all cells from top to bottom.

The notebook can also be executed from PowerShell with:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute Student_Performance_ML_Pipeline.ipynb --output Student_Performance_ML_Pipeline_executed.ipynb
```

## Expected Input and Output

The raw input dataset consists of twelve student features:
`Age`, `Gender`, `Ethnicity`, `ParentalEducation`, `StudyTimeWeekly`, `Absences`, `Tutoring`, `ParentalSupport`, `Extracurricular`, `Sports`, `Music`, and `Volunteering`.

The preprocessing stage (`ColumnTransformer`) transforms these 12 features into **29 complete processed features**:
- 3 numeric columns: `Age`, `StudyTimeWeekly`, `Absences` (median imputed)
- 26 one-hot encoded binary columns across categorical features: `Gender` (2), `Ethnicity` (4), `ParentalEducation` (5), `Tutoring` (2), `ParentalSupport` (5), `Extracurricular` (2), `Sports` (2), `Music` (2), `Volunteering` (2).

The Decision Tree model and `predict_student_performance` function directly accept the **complete processed data (29 columns)**. The notebook outputs the predicted numeric `GradeClass` label (`0 = A`, `1 = B`, `2 = C`, `3 = D`, `4 = F`), performance letter category, evaluation metrics, confusion matrix, tree visualization, feature importances, and saved `.bin` model files.

## Project Workflow

1. Load and inspect the CSV dataset.
2. Check data quality and target distribution.
3. Preserve original data and create a processed copy (`StudentID` dropped, `GPA` excluded to prevent leakage).
4. Build numeric and categorical preprocessing transformers.
5. Explore the data with documented visualizations.
6. Perform a stratified train-test split on raw features.
7. Fit the preprocessor on training data and transform train & test sets into complete processed data (29 features).
8. Train a Decision Tree Classifier (`criterion='gini'`, `max_depth=5`) directly on the 29 processed features.
9. Evaluate model performance (accuracy, weighted precision/recall/F1, classification report, confusion matrix).
10. Visualize the tree and feature importances with all 29 transformed feature names.
11. Predict student performance using `predict_student_performance(preprocessed_data)` (accepts 29 columns).
12. Serialize the trained 29-feature Decision Tree model (`student_performance_decision_tree_pipeline.bin`), preprocessor (`student_performance_preprocessor.bin`), and end-to-end pipeline (`student_performance_end_to_end_pipeline.bin`).
13. Demonstrate verified model loading and inference in `model_loading.ipynb`.