import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix,
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns

# 1. Завантажити датасет
df = pd.read_csv('dataset3_l4.csv')

# 2. Інформація з датасету
fieldsnum = len(df.columns)
recordnum = len(df)
print(f"Кількість записів: {recordnum}")
print(f"Кількість полів: {fieldsnum}")

print(df.head())

# 3.Поля датасету
fields = df.columns.tolist()
print(fields)
    
# 4. Перемішування функцією шафлспліт
n_splits = int(input("Введіть кількість варіантів перемішування (не менше трьох): "))


rs = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
splits = list(rs.split(df))

# Використовуємо другий варіант перемішування
train_index, test_index = splits[1]
train_df = df.iloc[train_index]
test_df = df.iloc[test_index]

# Перевірка збалансованості
train_balance = train_df['NObeyesdad'].value_counts(normalize=True)
print("Збалансованість навчальної вибірки:")
print(train_balance)


test_balance = test_df['NObeyesdad'].value_counts(normalize=True)
print("\nЗбалансованість тестової вибірки:")
print(test_balance)

# 5. Розділяємо за цільовою характеристикою 
X_train = train_df.drop(columns='NObeyesdad')
y_train = train_df['NObeyesdad']
X_test = test_df.drop(columns='NObeyesdad')
y_test = test_df['NObeyesdad']

categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numeric_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

encoder = OneHotEncoder()
X_train_categorical_encoded = encoder.fit_transform(X_train[categorical_columns])
X_test_categorical_encoded = encoder.transform(X_test[categorical_columns])

X_train_encoded = sp.hstack((X_train_categorical_encoded, sp.csr_matrix(X_train[numeric_columns].values)), format='csr')
X_test_encoded = sp.hstack((X_test_categorical_encoded, sp.csr_matrix(X_test[numeric_columns].values)), format='csr')

# Модель
knn = KNeighborsClassifier()
knn.fit(X_train_encoded, y_train)

# 6. Метрики
def metrics(trues, preds):
    metrics_results = {}
    metrics_results['Accuracy'] = accuracy_score(trues, preds)
    metrics_results['Precision'] = precision_score(trues, preds, average='weighted', zero_division=0)
    metrics_results['Recall'] = recall_score(trues, preds, average='weighted')
    metrics_results['F1 Score'] = f1_score(trues, preds, average='weighted')
    metrics_results['MCC'] = matthews_corrcoef(trues, preds)
    metrics_results['Balanced Accuracy'] = balanced_accuracy_score(trues, preds)
    return metrics_results

# обраховуємо метрики 
train_metrics = metrics(y_train, knn.predict(X_train_encoded))
test_metrics = metrics(y_test, knn.predict(X_test_encoded))

 
print("Train Metrics:")
for metric, value in train_metrics.items():
    print(f"{metric}: {value}")

print("\nTest Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value}")
    
# матриця помилок
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

plot_confusion_matrix(y_test, knn.predict(X_test_encoded), classes=np.unique(y_test), title='Confusion Matrix')

# графік метрики
def plot_metrics(metric_names, metric_values, title):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(metric_names)), metric_values, align='center', alpha=0.7)
    plt.xticks(range(len(metric_names)), metric_names)
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Гістограма метрик
plot_metrics(['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC', 'Balanced Accuracy'], 
             [test_metrics['Accuracy'], test_metrics['Precision'], test_metrics['Recall'], 
              test_metrics['F1 Score'], test_metrics['MCC'], test_metrics['Balanced Accuracy']],
             'Test Metrics')

# 7. Метрики в залежності від Мінковського (від 1 до 20)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Конвертація матриці в щільну
X_train_encoded_dense = X_train_encoded.toarray()
X_test_encoded_dense = X_test_encoded.toarray()


p_values = list(range(1, 21))
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
balanced_accuracy_scores = []

for p in p_values:
    knn = KNeighborsClassifier(n_neighbors=5, p=p)
    knn.fit(X_train_encoded_dense, y_train_encoded)
    y_pred = knn.predict(X_test_encoded_dense)
    
    accuracy_scores.append(accuracy_score(y_test_encoded, y_pred))
    precision_scores.append(precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0))
    recall_scores.append(recall_score(y_test_encoded, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test_encoded, y_pred, average='weighted'))
    balanced_accuracy_scores.append(balanced_accuracy_score(y_test_encoded, y_pred))

# створення графіків
def plot_metrics_vs_p(p_values, accuracy_scores, precision_scores, recall_scores, f1_scores, balanced_accuracy_scores):
    plt.figure(figsize=(12, 8))
    plt.plot(p_values, accuracy_scores, marker='o', label='Accuracy', color='blue')
    plt.plot(p_values, precision_scores, marker='o', label='Precision', color='green')
    plt.plot(p_values, recall_scores, marker='o', label='Recall', color='red')
    plt.plot(p_values, f1_scores, marker='o', label='F1 Score', color='purple')
    plt.plot(p_values, balanced_accuracy_scores, marker='o', label='Balanced Accuracy', color='orange')
    
    plt.title('Metrics vs Degree of Minkowski Metric')
    plt.xlabel('Degree of Minkowski Metric (p)')
    plt.ylabel('Score')
    plt.xticks(p_values) 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ГРафік метрики в залежності від мінковського
plot_metrics_vs_p(p_values, accuracy_scores, precision_scores, recall_scores, f1_scores, balanced_accuracy_scores)
