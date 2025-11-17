import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

# ULTRA-CONSERVATIVE APPROACH FOR 32% ACCURACY ISSUE
print("🛠️ SIMPLE & ROBUST CATTLE DISEASE TRAINER")
print("🎯 Designed to fix 32% accuracy issue")
print("="*50)

# Conservative parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 8          # Very small batch size
LEARNING_RATE = 0.00005 # Very low learning rate
EPOCHS = 50             # More epochs with early stopping

def create_conservative_generators():
    """Create very conservative data generators"""

    print("\n📊 Creating conservative data generators...")

    # MINIMAL augmentation to avoid overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,         # Minimal rotation
        width_shift_range=0.05,   # Minimal shift  
        height_shift_range=0.05,
        horizontal_flip=True,     # Only horizontal flip
        validation_split=0.25     # Larger validation set
    )

    # No augmentation for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.25
    )

    train_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    validation_generator = val_datagen.flow_from_directory(
        'dataset',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    print(f"   ✅ Training samples: {train_generator.samples}")
    print(f"   ✅ Validation samples: {validation_generator.samples}")
    print(f"   ✅ Classes found: {list(train_generator.class_indices.keys())}")

    return train_generator, validation_generator

def create_simple_robust_model():
    """Create simple, robust model that should work"""

    print("\n🏗️ Creating simple & robust model...")

    # Use VGG16 (proven, reliable architecture)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    # Keep base model frozen initially
    base_model.trainable = False

    # Very simple classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')  # 4 classes
    ])

    print(f"   ✅ Architecture: VGG16 + Simple Head")
    print(f"   ✅ Base model frozen: Yes")
    print(f"   ✅ Parameters: {model.count_params():,}")

    return model

def train_with_monitoring(model, train_gen, val_gen):
    """Train with extensive monitoring"""

    print("\n🎯 Starting conservative training...")

    # Compile with very conservative settings
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Conservative callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,   # More aggressive LR reduction
            patience=7,
            min_lr=1e-8,
            verbose=1
        )
    ]

    print(f"   🔧 Learning rate: {LEARNING_RATE}")
    print(f"   🔧 Batch size: {BATCH_SIZE}")
    print(f"   🔧 Max epochs: {EPOCHS}")
    print(f"   🔧 Early stopping patience: 15")

    # Train the model
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    return history

def evaluate_and_diagnose(model, val_gen, history):
    """Comprehensive evaluation and diagnosis"""

    print("\n📊 Evaluating model performance...")

    # Get final accuracy
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    final_val_acc = history.history['val_accuracy'][-1]

    print(f"\n📈 TRAINING RESULTS:")
    print(f"   Best validation accuracy: {best_val_acc:.1%} (epoch {best_epoch})")
    print(f"   Final validation accuracy: {final_val_acc:.1%}")

    # Generate predictions
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    print(f"\n📊 CONFUSION MATRIX:")
    print("   ", end="")
    for label in class_labels:
        print(f"{label[:8]:>10}", end="")
    print()

    for i, true_label in enumerate(class_labels):
        print(f"{true_label[:8]:>8}:", end="")
        for j in range(len(class_labels)):
            print(f"{cm[i,j]:10d}", end="")
        print()

    # Detailed analysis
    print(f"\n🔍 DETAILED ANALYSIS:")

    total_samples = len(predicted_classes)
    prediction_distribution = {}

    for i, class_name in enumerate(class_labels):
        true_count = sum(true_classes == i)
        pred_count = sum(predicted_classes == i)
        correct_count = cm[i, i]

        recall = correct_count / true_count if true_count > 0 else 0
        precision = correct_count / pred_count if pred_count > 0 else 0

        prediction_distribution[class_name] = pred_count

        print(f"\n   {class_name}:")
        print(f"      True samples: {true_count}")
        print(f"      Predictions: {pred_count} ({pred_count/total_samples*100:.1f}%)")
        print(f"      Correct: {correct_count}")
        print(f"      Recall: {recall:.1%}")
        print(f"      Precision: {precision:.1%}")

    # Check for bias issues
    print(f"\n🚨 BIAS ANALYSIS:")
    max_predictions = max(prediction_distribution.values())
    dominant_class = max(prediction_distribution, key=prediction_distribution.get)
    bias_percentage = (max_predictions / total_samples) * 100

    if bias_percentage > 80:
        print(f"   🚨 SEVERE BIAS: {dominant_class} gets {bias_percentage:.1f}% of predictions")
        print(f"   💡 Model is biased toward majority class")
        bias_level = "severe"
    elif bias_percentage > 60:
        print(f"   ⚠️  MODERATE BIAS: {dominant_class} gets {bias_percentage:.1f}% of predictions")
        bias_level = "moderate"  
    else:
        print(f"   ✅ BALANCED: Predictions reasonably distributed")
        bias_level = "none"

    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")

    if best_val_acc >= 0.80:
        assessment = "excellent"
        print(f"   ✅ EXCELLENT: {best_val_acc:.1%} accuracy achieved!")
    elif best_val_acc >= 0.65:
        assessment = "good"
        print(f"   ✅ GOOD: {best_val_acc:.1%} accuracy (significant improvement)")
    elif best_val_acc >= 0.45:
        assessment = "moderate"
        print(f"   ⚠️  MODERATE: {best_val_acc:.1%} accuracy (some improvement)")
    else:
        assessment = "poor"
        print(f"   ❌ POOR: {best_val_acc:.1%} accuracy (still problematic)")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if assessment == "poor":
        print("   🚨 CRITICAL DATA ISSUES:")
        print("      1. Images may be mislabeled")
        print("      2. Disease symptoms not visible")
        print("      3. Need professional veterinary dataset")
        print("      4. Consider different problem formulation")
    elif bias_level == "severe":
        print("   ⚖️  BIAS ISSUES:")
        print("      1. Increase class weights")
        print("      2. Balance dataset further")  
        print("      3. Use focal loss")
        print("      4. Collect more minority class data")
    elif assessment == "moderate":
        print("   📈 IMPROVEMENT POSSIBLE:")
        print("      1. Train longer (more epochs)")
        print("      2. Fine-tune hyperparameters")
        print("      3. Try data augmentation")
        print("      4. Unfreeze some base layers")
    else:
        print("   🎉 MODEL WORKING WELL!")
        print("      1. Save and deploy model")
        print("      2. Test on new data")
        print("      3. Monitor real-world performance")

    return best_val_acc, assessment, bias_level

def plot_training_history(history):
    """Plot training history"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simple_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""

    print("🚀 STARTING SIMPLE & ROBUST TRAINING")
    print("   Designed to fix 32% accuracy issue")
    print("   Using conservative, proven approach")

    # Create data generators
    train_gen, val_gen = create_conservative_generators()

    # Create model
    model = create_simple_robust_model()

    # Train model
    history = train_with_monitoring(model, train_gen, val_gen)

    # Plot results
    plot_training_history(history)

    # Evaluate and diagnose
    final_acc, assessment, bias_level = evaluate_and_diagnose(model, val_gen, history)

    # Save model if decent
    if final_acc > 0.5:  # Better than random
        print("\n💾 Saving improved model...")
        model.save('cattle_disease_model_simple.h5')

        # Save class indices
        class_indices = train_gen.class_indices
        with open('class_indices.json', 'w') as f:
            json.dump(class_indices, f, indent=2)

        print("   ✅ Model saved as 'cattle_disease_model_simple.h5'")
    else:
        print("\n❌ Model performance too poor to save")
        print("   💡 Consider using diagnostic_system.py for deeper analysis")

    print("\n" + "="*50)
    print("🏁 SIMPLE TRAINING COMPLETE")
    print(f"📊 Final Result: {final_acc:.1%} accuracy")
    print(f"📈 Assessment: {assessment.upper()}")

    if final_acc > 0.65:
        print("✅ SUCCESS: Significant improvement achieved!")
    elif final_acc > 0.45:
        print("⚠️  PARTIAL: Some improvement, needs more work")
    else:
        print("❌ FAILED: Fundamental data issues remain")
        print("   💡 Run diagnostic_system.py for detailed analysis")

    print("="*50)

if __name__ == "__main__":
    main()