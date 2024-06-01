import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load preprocessed data and model
valid_set = joblib.load("valid_set.pkl")
model = load_model('best_model.h5')

# Evaluate model
val_loss, val_acc = model.evaluate(valid_set)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

# Generate predictions and classification report
y_true = valid_set.classes
y_pred = model.predict(valid_set)
y_pred_classes = y_pred.argmax(axis=-1)
class_names = list(valid_set.class_indices.keys())

cr = classification_report(y_true, y_pred_classes, target_names=class_names)
print('Classification report:\n', cr)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()