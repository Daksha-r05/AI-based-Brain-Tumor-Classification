import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set the path to the dataset - UPDATED TO YOUR ACTUAL DATASET PATH
dataset_path = "/Users/sriranga/Desktop/myrealel for jupyter maybe"

# Define the training and testing directories
train_dir = os.path.join(dataset_path, "Training")
test_dir = os.path.join(dataset_path, "Testing")

# Define the categories
categories = ["glioma", "meningioma", "notumor", "pituitary"]

# Set the image size
image_size = (150, 150)
batch_size = 32

# Load the pre-trained model
model_path = '/Users/sriranga/Desktop/brain_tumor_flask_app/brain_tumor_detection_model.keras'

try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please make sure the model file exists at the specified path.")
    exit()

# Check if dataset directories exist
if not os.path.exists(test_dir):
    print(f"❌ Test directory not found: {test_dir}")
    print("Please update the dataset_path variable to point to your dataset location.")
    exit()

print(f"✅ Found dataset at: {dataset_path}")
print(f"✅ Test directory: {test_dir}")

# Data preprocessing for testing
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

print(f"📊 Found {test_generator.samples} test images")
print(f"📋 Class indices: {test_generator.class_indices}")

# Make predictions on the test dataset
print("🤖 Making predictions on your actual dataset...")
predictions = model.predict(test_generator, verbose=1)
predicted_categories = np.argmax(predictions, axis=1)
true_categories = test_generator.classes

# Create confusion matrix
print("📈 Generating real confusion matrix from your data...")
cm = confusion_matrix(true_categories, predicted_categories)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))

# Create a more detailed confusion matrix plot
plt.subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=categories, yticklabels=categories)
plt.title("Real Confusion Matrix\n(Brain Tumor Detection)", fontsize=14, pad=20)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Calculate and display metrics
precision = np.diag(cm) / np.sum(cm, axis=0)
recall = np.diag(cm) / np.sum(cm, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

# Display metrics as text
plt.subplot(2, 2, 2)
plt.axis('off')
metrics_text = f"""
YOUR MODEL'S REAL METRICS:
Overall Accuracy: {accuracy:.3f}

PER-CLASS METRICS:
"""
for i, category in enumerate(categories):
    metrics_text += f"\n{category.upper()}:"
    metrics_text += f"\n  Precision: {precision[i]:.3f}"
    metrics_text += f"\n  Recall: {recall[i]:.3f}"
    metrics_text += f"\n  F1-Score: {f1_score[i]:.3f}"

plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='top', fontfamily='monospace')

# Create a bar plot of per-class accuracy
plt.subplot(2, 2, 3)
class_accuracy = recall  # Recall is the same as per-class accuracy
bars = plt.bar(categories, class_accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.title("Per-Class Accuracy", fontsize=14)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1)
for bar, acc in zip(bars, class_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom')

# Create a pie chart of dataset distribution
plt.subplot(2, 2, 4)
class_counts = np.sum(cm, axis=1)  # Count of samples per class
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
plt.pie(class_counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title("Test Dataset Distribution", fontsize=14)

plt.tight_layout()
plt.savefig('brain_tumor_confusion_matrix_real.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed classification report
print("\n" + "="*60)
print("YOUR MODEL'S CLASSIFICATION REPORT")
print("="*60)
print(classification_report(true_categories, predicted_categories, target_names=categories))

print("\n" + "="*60)
print("DETAILED METRICS BY CLASS")
print("="*60)
for i, category in enumerate(categories):
    print(f"\n🏥 Class: {category.upper()}")
    print(f"   Precision: {precision[i]:.4f}")
    print(f"   Recall: {recall[i]:.4f}")
    print(f"   F1-Score: {f1_score[i]:.4f}")
    print(f"   Samples: {class_counts[i]}")

print(f"\n🎯 Overall Accuracy: {accuracy:.4f}")

# Additional analysis
print("\n" + "="*60)
print("MODEL ANALYSIS")
print("="*60)
print("✅ Model Architecture: CNN with 4 classes")
print("✅ Input Shape:", model.input_shape)
print("✅ Output Shape:", model.output_shape)
print("✅ Total Parameters:", model.count_params())

# Plot sample predictions
print("\n🖼️ Generating sample prediction visualization...")
test_images = test_generator.filenames
sample_indices = np.random.choice(range(len(test_images)), size=9, replace=False)
sample_images = [test_images[i] for i in sample_indices]
sample_predictions = [categories[predicted_categories[i]] for i in sample_indices]
sample_true_labels = [categories[true_categories[i]] for i in sample_indices]

plt.figure(figsize=(15, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img_path = os.path.join(test_dir, sample_images[i])
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        plt.imshow(img)
        if sample_predictions[i] == sample_true_labels[i]:
            plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}", 
                     color='green', fontsize=10)
        else:
            plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}", 
                     color='red', fontsize=10)
        plt.axis("off")
    else:
        plt.text(0.5, 0.5, f"Image not found:\n{sample_images[i]}", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis("off")

plt.suptitle("Sample Predictions from Your Dataset (Green=Correct, Red=Incorrect)", fontsize=16)
plt.tight_layout()
plt.savefig('sample_predictions_real.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the confusion matrix data
confusion_data = {
    'confusion_matrix': cm.tolist(),
    'categories': categories,
    'metrics': {
        'accuracy': float(accuracy),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1_score.tolist(),
        'class_counts': class_counts.tolist()
    },
    'dataset_info': {
        'test_samples': int(test_generator.samples),
        'dataset_path': dataset_path
    }
}

import json
with open('confusion_matrix_real_data.json', 'w') as f:
    json.dump(confusion_data, f, indent=2)

print("\n💾 Files saved:")
print("   - brain_tumor_confusion_matrix_real.png")
print("   - sample_predictions_real.png")
print("   - confusion_matrix_real_data.json")

print("\n🎉 This is your REAL confusion matrix based on your actual dataset and model!") 