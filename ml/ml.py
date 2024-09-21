import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Load data from the CSV file
url = "../databases/2023-ByDay.csv"
df = pd.read_csv(url, delimiter=';')

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix with a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Prepare the data
X = df[['Avg_Temperature', 'Max_Humidity', 'Max_Preciptation']]
y = df['Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

# Creating folds for cross-validation
num_particoes = 10  # number of folds for cross-validation
kfold = KFold(n_splits=num_particoes, shuffle=True, random_state=7)

# Setting a global seed for reproducibility
np.random.seed(7)

# Lists to store models, results, and model names
models = []
results = []
names = []

# Preparing models
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('LR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('SVM', SVR()))

# Evaluating each model
for name, model in models:
    # Create a pipeline with standardization
    pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', model)])

    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)

    # Print MSE, standard deviation of MSE, and RMSE of the 10 cross-validation results
    msg = "%s: MSE %0.2f (%0.2f) - RMSE %0.2f" % (name, abs(cv_results.mean()), cv_results.std(), np.sqrt(abs(cv_results.mean())))
    print(msg)

# Boxplot comparing the models
fig = plt.figure()
fig.suptitle('Comparação do MSE dos Modelos')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Selecting the best model (the one with the lowest MSE)
best_index = np.argmin([abs(cv.mean()) for cv in results])
best_model_name = names[best_index]
best_model = models[best_index][1]

# Training the best model with all training data
best_pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', best_model)])
best_pipeline.fit(X_train, y_train)

# Evaluation of the final model on the test data
y_pred = best_pipeline.predict(X_test)
print(f"Avaliação do modelo {best_model_name}:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")

# Exporting the trained model
joblib.dump(best_pipeline, '../predictor/model.pkl')
print("Modelo exportado para '../predictor/model.pkl'.")
