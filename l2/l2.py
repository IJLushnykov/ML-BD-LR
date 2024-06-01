import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve, precision_recall_curve,
    auc, balanced_accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# 1
data = pd.read_csv('KM-11-3.csv')

# Виведення перших декількох рядків даних для огляду
print(data.head())

# 2
print("Кількість об'єктів кожного класу:")
print(data['GT'].value_counts())

data['Model_1_1'] = 1 - data['Model_1_0']

# 3
def get_metrics(data):
  # Визначення метрик для різних порогів
  thresholds = np.linspace(0, 1, 101)
  metrics_results_1 = {
      'Number': [],
      'Threshold': [],
      'Accuracy': [],
      'Precision': [],
      'Recall': [],
      'F1 Score': [],
      'MCC': [],
      'Balanced Accuracy': [],
      'AUC-PR': None,
      'AUC-ROC': None,
      'Youden': []
  }
  metrics_results_2 = {
      'Number': [],
      'Threshold': [],
      'Accuracy': [],
      'Precision': [],
      'Recall': [],
      'F1 Score': [],
      'MCC': [],
      'Balanced Accuracy': [],
      'AUC-PR': None,
      'AUC-ROC': None,
      'Youden': []
  }

  for threshold in thresholds:
      pred_model1 = (data['Model_1_1'] >= threshold).astype(int)
      pred_model2 = (data['Model_2_1'] >= threshold).astype(int)

      # Використання моделі 1
      for metrics_results, pred_model in zip([metrics_results_1, metrics_results_2], [pred_model1, pred_model2]):
        metrics_results['Number'].append(sum(pred_model))
        metrics_results['Threshold'].append(threshold)
        metrics_results['Accuracy'].append(accuracy_score(data['GT'], pred_model))
        metrics_results['Precision'].append(precision_score(data['GT'], pred_model, zero_division=0))
        metrics_results['Recall'].append(recall_score(data['GT'], pred_model))
        metrics_results['F1 Score'].append(f1_score(data['GT'], pred_model))
        metrics_results['MCC'].append(matthews_corrcoef(data['GT'], pred_model))
        metrics_results['Balanced Accuracy'].append(balanced_accuracy_score(data['GT'], pred_model))
        metrics_results['Youden'].append(recall_score(data['GT'], pred_model) + recall_score(data['GT'], pred_model, pos_label=0) - 1)

  for metrics_results, model in zip([metrics_results_1, metrics_results_2], [data['Model_1_1'], data['Model_2_1']]):
    precision, recall, _ = precision_recall_curve(data['GT'], model)
    metrics_results['AUC-PR'] = auc(recall, precision)
    fpr, tpr, _ = roc_curve(data['GT'], model)
    metrics_results['AUC-ROC'] = auc(fpr, tpr)
    metrics_results['Youden'] = max(metrics_results['Youden'])

  return pd.DataFrame(metrics_results_1), pd.DataFrame(metrics_results_2)

metrics_results_1, metrics_results_2 = get_metrics(data)
print('Обраховані метрики')
print('Модель 1')
print(metrics_results_1)
print('Модель 2')
print(metrics_results_2)

def plot_metric_graph(metrics, name):
  for k, m in enumerate(metrics):

    fig, ax = plt.subplots()
    metrics = list(m.columns)
    xs = m[name]
    metrics.remove('Threshold')
    metrics.remove('Number')
    color_map = plt.cm.get_cmap('tab10', len(metrics))
    for n, i in enumerate(metrics):
      ys = m[i]
      color = color_map(n)
      if len(np.unique(ys)) > 1:
        plt.plot(xs, ys, label=i, color=color)
        if name == 'Threshold':
          plt.plot([-0.1, 1.1], [np.max(ys)]*2, '--', color=color)
        elif name == 'Number':
          plt.axvline(xs[np.argmax(ys)], linestyle=':', color=color)
    ax.set_xlabel(name)
    ax.set_ylabel('Values')
    plt.legend()
    plt.title(f'Model {k+1}')
    plt.show()



def find_intersection(x1, y1, x2, y2):
    f1 = interp1d(x1, y1)
    f2 = interp1d(x2, y2)
    intersection_x = brentq(lambda x: f1(x) - f2(x), 0, 1)
    intersection_y = f1(intersection_x)
    return intersection_x, intersection_y

def plot_metrics(df, model_name):
    # Визначення метрик та порогів
    precision, recall, _ = precision_recall_curve(df['GT'], df[model_name])
    fpr, tpr, _ = roc_curve(df['GT'], df[model_name])

    # Обчислення перетину для PR кривої
    pr_intersection_x, pr_intersection_y = find_intersection(recall, precision, [0, 1], [0, 1])

    # Візуалізація PR кривої
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label='PR Curve')
    plt.plot([0, 1], [np.mean(df['GT'] == 1), np.mean(df['GT'] == 1)], 'b:')
    plt.plot([0, 1], [0, 1], 'k:')
    plt.scatter(pr_intersection_x, pr_intersection_y, color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend()

    # Обчислення та візуалізація ROC кривої
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k:')
    optimal_idx = np.argmin(np.abs(tpr + fpr - 1))
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend()
    plt.show()



def plot_pr_roc(data):
  models = ['Model_1_1', 'Model_2_1']
  for m in models:
    print(f"Metrics for {m[:-2]}")
    plot_metrics(data, m)

# 3.b
plot_metric_graph([metrics_results_1, metrics_results_2], 'Threshold')
# 3.c
plot_metric_graph([metrics_results_1, metrics_results_2], 'Number')
# 3.d
plot_pr_roc(data)

# 4
#Визначення яка модель краще

# 5
# Визначення параметру K
birthday = "07-07"
month = int(birthday.split("-")[1])  # місяць народження
K = month % 4

# 6
# Визначення відсотку видалення
percentage_to_remove = 50 + 10 * K


# Видалення вказаного відсотка об'єктів класу 1
class_1_data = data[data['GT'] == 1]
remove_n = int(len(class_1_data) * (percentage_to_remove / 100))
print(f"Відсоток видалення об'єктів класу 1: {round(100 * remove_n/len(class_1_data), 2)}%")
removed_class_1_data = class_1_data.sample(n=remove_n, random_state=1)  # використання random_state для відтворюваності

# Оновлення набору даних
new_data = pd.concat([data[data['GT'] == 0], class_1_data.drop(removed_class_1_data.index)])

# Виведення кількості елементів у кожному класі після видалення
print("Кількість елементів у кожному класі після видалення:")
print(new_data['GT'].value_counts())

# 7
metrics_results_1, metrics_results_2 = get_metrics(new_data)
print('Обраховані метрики')
print('Модель 1')
print(metrics_results_1)
print('Модель 2')
print(metrics_results_2)

# 7.b
plot_metric_graph([metrics_results_1, metrics_results_2], 'Threshold')
# 7.c
plot_metric_graph([metrics_results_1, metrics_results_2], 'Number')
# 7.d
plot_pr_roc(new_data)

# 8, 9
# Визначити яка модель краще, яка стабільніша до перекосів
