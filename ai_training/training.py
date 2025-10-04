# write a script for training a neural network to classify activities based on imu sensor data
# dataset: ai_training/dataset/our_dataset.csv
# data structure: Timestamp,Activity_ID,Activity_Type,Ax,Ay,Az,Gx,Gy,Gz
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json

# Configuration
class Config:
    # Data parameters
    DATASET_PATH = 'dataset'
    WINDOW_SIZE = 128  # Number of timesteps per window
    STEP_SIZE = 64  # Overlap of 50%
    SAMPLING_RATE = 50  # Hz (adjust based on your actual sampling rate)
    
    # Sensor columns (15 IMU features)
    SENSOR_COLUMNS = [
        'mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz',
        'adxl_ax', 'adxl_ay', 'adxl_az',
        'l3gd_gx', 'l3gd_gy', 'l3gd_gz',
        'lsm_ax', 'lsm_ay', 'lsm_az'
    ]
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 15  # Early stopping patience
    
    # Model parameters
    USE_AUGMENTATION = True
    DROPOUT_RATE = 0.3


class IMUDataLoader:
    """Loads and preprocesses IMU data from CSV files"""
    
    def __init__(self, config: Config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def extract_activity_segments(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, str]]:
        """
        Extract activity segments based on joystick button presses
        Returns list of (segment_df, activity_type) tuples
        """
        segments = []
        
        # Find button press events (assuming 1 = pressed)
        left_presses = df[df['joy_lb'] == 1].index.tolist()
        right_presses = df[df['joy_rb'] == 1].index.tolist()
        
        # Match start (left) and end (right) buttons
        # Simple approach: pair each left press with the next right press
        for i, start_idx in enumerate(left_presses):
            # Find the next right press after this left press
            end_candidates = [idx for idx in right_presses if idx > start_idx]
            
            if end_candidates:
                end_idx = end_candidates[0]
                segment = df.iloc[start_idx:end_idx + 1].copy()
                activity_type = segment['Activity_Type'].iloc[0]
                
                # Only include if segment is long enough
                if len(segment) >= self.config.WINDOW_SIZE:
                    segments.append((segment, activity_type))
        
        return segments
    
    def create_windows(self, segment: pd.DataFrame, activity_type: str) -> Tuple[np.ndarray, List[str]]:
        """
        Create sliding windows from a segment
        Returns (windows, labels)
        """
        sensor_data = segment[self.config.SENSOR_COLUMNS].values
        windows = []
        labels = []
        
        for i in range(0, len(sensor_data) - self.config.WINDOW_SIZE + 1, self.config.STEP_SIZE):
            window = sensor_data[i:i + self.config.WINDOW_SIZE]
            windows.append(window)
            labels.append(activity_type)
        
        return np.array(windows), labels
    
    def load_csv_files(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all CSV files from a split directory (train/val/test)
        Returns (X, y) where X is windows and y is labels
        """
        split_path = os.path.join(self.config.DATASET_PATH, split)
        
        if not os.path.exists(split_path):
            raise ValueError(f"Split directory not found: {split_path}")
        
        all_windows = []
        all_labels = []
        
        csv_files = [f for f in os.listdir(split_path) if f.endswith('.csv')]
        print(f"\nLoading {len(csv_files)} CSV files from {split} split...")
        
        for csv_file in csv_files:
            file_path = os.path.join(split_path, csv_file)
            print(f"  Processing: {csv_file}")
            
            try:
                df = pd.read_csv(file_path)
                
                # Validate required columns
                required_cols = self.config.SENSOR_COLUMNS + ['Activity_Type', 'joy_lb', 'joy_rb']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"    Warning: Missing columns {missing_cols}, skipping file")
                    continue
                
                # Extract activity segments
                segments = self.extract_activity_segments(df)
                print(f"    Found {len(segments)} activity segments")
                
                # Create windows from each segment
                for segment_df, activity_type in segments:
                    windows, labels = self.create_windows(segment_df, activity_type)
                    all_windows.append(windows)
                    all_labels.extend(labels)
                    
            except Exception as e:
                print(f"    Error processing {csv_file}: {e}")
                continue
        
        if not all_windows:
            raise ValueError(f"No valid data found in {split} split")
        
        X = np.concatenate(all_windows, axis=0)
        y = np.array(all_labels)
        
        print(f"\nLoaded {len(X)} windows from {split} split")
        print(f"Window shape: {X.shape}")
        print(f"Unique activities: {np.unique(y)}")
        
        return X, y
    
    def load_all_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load train, validation, and test splits"""
        data = {}
        
        for split in ['train', 'val', 'test']:
            try:
                X, y = self.load_csv_files(split)
                
                # Fit scaler and label encoder on training data only
                if split == 'train':
                    X_reshaped = X.reshape(-1, X.shape[-1])
                    self.scaler.fit(X_reshaped)
                    self.label_encoder.fit(y)
                
                # Transform data
                X_normalized = self.normalize_windows(X)
                y_encoded = self.label_encoder.transform(y)
                
                data[split] = (X_normalized, y_encoded)
                
            except Exception as e:
                print(f"Warning: Could not load {split} split: {e}")
                if split == 'train':
                    raise  # Training data is mandatory
        
        return data
    
    def normalize_windows(self, X: np.ndarray) -> np.ndarray:
        """Normalize windows using fitted scaler"""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.transform(X_reshaped)
        return X_normalized.reshape(original_shape)


class DataAugmenter:
    """Augment IMU data for better generalization"""
    
    @staticmethod
    def add_noise(X: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to sensor readings"""
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    @staticmethod
    def scale_magnitude(X: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Scale the magnitude of sensor readings"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return X * scale
    
    @staticmethod
    def time_warp(X: np.ndarray, warp_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply time warping by resampling"""
        warp_factor = np.random.uniform(warp_range[0], warp_range[1])
        new_length = int(X.shape[0] * warp_factor)
        
        # Resample each feature independently
        warped = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[1]):
            old_indices = np.linspace(0, len(X) - 1, new_length)
            new_indices = np.linspace(0, len(X) - 1, X.shape[0])
            warped[:, i] = np.interp(new_indices, old_indices, 
                                     np.interp(old_indices, range(len(X)), X[:, i]))
        
        return warped
    
    @staticmethod
    def augment_batch(X: np.ndarray, y: np.ndarray, augmentation_prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentation to a batch"""
        augmented_X = []
        
        for window in X:
            if np.random.random() < augmentation_prob:
                # Randomly choose augmentation
                aug_type = np.random.choice(['noise', 'scale', 'warp'])
                
                if aug_type == 'noise':
                    window = DataAugmenter.add_noise(window)
                elif aug_type == 'scale':
                    window = DataAugmenter.scale_magnitude(window)
                elif aug_type == 'warp':
                    window = DataAugmenter.time_warp(window)
            
            augmented_X.append(window)
        
        return np.array(augmented_X), y


def build_model(input_shape: Tuple[int, int], num_classes: int, config: Config) -> keras.Model:
    """
    Build a hybrid CNN-LSTM model for activity recognition
    
    Architecture:
    - Conv1D layers to extract local temporal features
    - LSTM to capture temporal dependencies
    - Dense layers for classification
    """
    inputs = keras.Input(shape=input_shape, name='imu_input')
    
    # Convolutional block 1
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    
    # Convolutional block 2
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    
    # Convolutional block 3
    x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    
    # LSTM layers
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='activity_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='imu_activity_classifier')
    
    return model


def plot_training_history(history: keras.callbacks.History, save_path: str = 'training_history.png'):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nTraining history plot saved to {save_path}")


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray, 
                   label_encoder: LabelEncoder):
    """Evaluate model and print detailed metrics"""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred
                                target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved to confusion_matrix.png")
    
    # Per-class accuracy
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for class_name, acc in zip(label_encoder.classes_, class_accuracies):
        print(f"{class_name:20s}: {acc*100:.2f}%")


def main():
    """Main training pipeline"""
    print("="*60)
    print("IMU Activity Recognition Training Pipeline")
    print("="*60)
    
    # Initialize configuration
    config = Config()
    
    # Load data
    print("\n[1/5] Loading and preprocessing data...")
    data_loader = IMUDataLoader(config)
    data = data_loader.load_all_splits()
    
    X_train, y_train = data['train']
    X_val, y_val = data.get('val', (None, None))
    X_test, y_test = data.get('test', (None, None))
    
    # Data summary
    num_classes = len(data_loader.label_encoder.classes_)
    print(f"\nData Summary:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {data_loader.label_encoder.classes_}")
    print(f"  Training samples: {len(X_train)}")
    if X_val is not None:
        print(f"  Validation samples: {len(X_val)}")
    if X_test is not None:
        print(f"  Test samples: {len(X_test)}")
    print(f"  Window shape: {X_train.shape[1:]}")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(y_train),
                                         y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Build model
    print("\n[2/5] Building model...")
    model = build_model(input_shape=X_train.shape[1:], 
                       num_classes=num_classes,
                       config=config)
    model.summary()
    
    # Compile model
    print("\n[3/5] Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n[4/5] Training model...")
    validation_data = (X_val, y_val) if X_val is not None else None
    
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=validation_data,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    if X_test is not None:
        print("\n[5/5] Evaluating on test set...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        
        evaluate_model(model, X_test, y_test, data_loader.label_encoder)
    
    # Save artifacts
    print("\n" + "="*60)
    print("Saving model and preprocessing objects...")
    model.save('final_model.keras')
    
    # Save label encoder and scaler
    artifacts = {
        'classes': data_loader.label_encoder.classes_.tolist(),
        'scaler_mean': data_loader.scaler.mean_.tolist(),
        'scaler_scale': data_loader.scaler.scale_.tolist(),
        'window_size': config.WINDOW_SIZE,
        'sensor_columns': config.SENSOR_COLUMNS
    }
    
    with open('model_artifacts.json', 'w') as f:
        json.dump(artifacts, f, indent=2)
    
    print("\nSaved files:")
    print("  - best_model.keras (best model during training)")
    print("  - final_model.keras (final model)")
    print("  - model_artifacts.json (preprocessing parameters)")
    print("  - training_history.png (training curves)")
    print("  - confusion_matrix.png (test set confusion matrix)")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()