import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt

# ==================== CONFIGURATION ====================
WINDOW_SIZE = 50  # Number of samples per window
WINDOW_STEP = 25  # Overlap between windows
AUGMENTATION_FACTOR = 3  # How many augmented samples per original

# Gesture to sensor mapping
GESTURE_SENSORS = {
    'Jumping': ['adxl_ax', 'adxl_ay', 'adxl_az'],  # ADXL345 accelerometer only
    'Kucania': ['mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz'],  # MPU6050 accel + gyro
    'Udarzenia': ['l3gd_gx', 'l3gd_gy', 'l3gd_gz', 'lsm_ax', 'lsm_ay', 'lsm_az'],  # L3GD20 gyro + LSM303 accel
    'Udarzenia_lewa': ['mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz'],  # MPU6050 accel + gyro
    'Strzal': ['adxl_ax', 'adxl_ay', 'adxl_az', 'lsm_ax', 'lsm_ay', 'lsm_az'],  # ADXL345 + LSM303 accelerometers
    'Syf': ['mpu_ax', 'mpu_ay', 'mpu_az', 'adxl_ax', 'adxl_ay', 'adxl_az']  # MPU + ADXL accelerometers
}

# ==================== DATA AUGMENTATION ====================
def add_noise(data, noise_level=0.05):
    """Add Gaussian noise to data"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise * np.std(data, axis=0)

def scale_data(data, scale_range=(0.9, 1.1)):
    """Random scaling augmentation"""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale

def time_warp(data, sigma=0.2):
    """Time warping augmentation"""
    num_samples = data.shape[0]
    warp = np.random.normal(1.0, sigma, num_samples)
    warp = np.cumsum(warp)
    warp = (warp - warp.min()) / (warp.max() - warp.min()) * (num_samples - 1)
    indices = np.clip(warp, 0, num_samples - 1).astype(int)
    return data[indices]

def augment_window(window, num_augmentations=3):
    """Apply multiple augmentation techniques"""
    augmented = [window]  # Original window
    
    for _ in range(num_augmentations):
        aug = window.copy()
        
        # Randomly apply augmentations
        if np.random.random() > 0.5:
            aug = add_noise(aug, noise_level=np.random.uniform(0.02, 0.08))
        if np.random.random() > 0.5:
            aug = scale_data(aug, scale_range=(0.85, 1.15))
        if np.random.random() > 0.5:
            aug = time_warp(aug, sigma=np.random.uniform(0.1, 0.3))
            
        augmented.append(aug)
    
    return augmented

# ==================== DATA LOADING ====================
def load_gesture_data(data_dir='dataset'):
    """Load all gesture CSV files"""
    all_data = []
    gesture_files = {
        'jumping': 'data_*_Jumping_*.csv',
        'kucanie': 'data_*_Kucanie_*.csv',
        'udarzanie': 'data_*_Udarzanie_*.csv',
        'udarzanie_l': 'data_*_Udarzanie_L_*.csv',
        'strzal': 'data_*_Strzal_*.csv',
        'syf': 'data_*_Syf_*.csv'
    }
    
    for gesture, pattern in gesture_files.items():
        files = glob.glob(os.path.join(data_dir, pattern))
        print(f"Loading {gesture}: found {len(files)} files")
        
        for file in files:
            df = pd.read_csv(file)
            df['gesture'] = gesture
            df['file'] = os.path.basename(file)
            all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No data files found in {data_dir}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal samples loaded: {len(combined_df)}")
    print(f"Gestures distribution:\n{combined_df['gesture'].value_counts()}")
    
    return combined_df

# ==================== SLIDING WINDOW EXTRACTION ====================
def create_sliding_windows(df, window_size=50, step_size=25):
    """Create sliding windows from time series data"""
    windows = []
    labels = []
    
    # Group by file to maintain temporal continuity
    for file in df['file'].unique():
        file_data = df[df['file'] == file].sort_values('Timestamp')
        gesture = file_data['gesture'].iloc[0]
        
        # Get relevant sensors for this gesture
        sensor_cols = GESTURE_SENSORS.get(gesture, GESTURE_SENSORS['syf'])
        
        # Extract sensor data
        sensor_data = file_data[sensor_cols].values
        
        # Create windows
        for i in range(0, len(sensor_data) - window_size + 1, step_size):
            window = sensor_data[i:i + window_size]
            
            if window.shape[0] == window_size:  # Ensure full window
                windows.append(window)
                labels.append(gesture)
    
    return np.array(windows), np.array(labels)

# ==================== FEATURE EXTRACTION ====================
def extract_features(windows):
    """Extract statistical features from windows"""
    features = []
    
    for window in windows:
        # Calculate statistical features for each sensor axis
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        min_val = np.min(window, axis=0)
        max_val = np.max(window, axis=0)
        median = np.median(window, axis=0)
        
        # Combine features
        window_features = np.concatenate([mean, std, min_val, max_val, median])
        features.append(window_features)
    
    return np.array(features)

# ==================== MODEL ARCHITECTURE ====================
def create_lstm_model(input_shape, num_classes):
    """Create LSTM-based model for time series classification"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First LSTM layer
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        # Second LSTM layer
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        # Third LSTM layer
        layers.LSTM(32),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_cnn_lstm_model(input_shape, num_classes):
    """Create hybrid CNN-LSTM model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # CNN layers for feature extraction
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # LSTM layers for temporal patterns
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        
        layers.LSTM(32),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# ==================== MAIN TRAINING PIPELINE ====================
def main():
    print("=" * 60)
    print("GESTURE RECOGNITION NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_gesture_data('dataset')
    
    # 2. Create sliding windows
    print("\n[2/6] Creating sliding windows...")
    windows, labels = create_sliding_windows(df, WINDOW_SIZE, WINDOW_STEP)
    print(f"Created {len(windows)} windows")
    
    # 3. Data augmentation
    print("\n[3/6] Applying data augmentation...")
    augmented_windows = []
    augmented_labels = []
    
    for window, label in zip(windows, labels):
        aug_windows = augment_window(window, AUGMENTATION_FACTOR)
        augmented_windows.extend(aug_windows)
        augmented_labels.extend([label] * len(aug_windows))
    
    X = np.array(augmented_windows)
    y = np.array(augmented_labels)
    
    print(f"After augmentation: {len(X)} samples")
    
    # 4. Encode labels
    print("\n[4/6] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = keras.utils.to_categorical(y_encoded)
    
    print(f"Classes: {label_encoder.classes_}")
    
    # 5. Normalize data
    print("\n[5/6] Normalizing data...")
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # 6. Build and train model
    print("\n[6/6] Building and training model...")
    
    # Try both models
    model = create_cnn_lstm_model(
        input_shape=(n_timesteps, n_features),
        num_classes=len(label_encoder.classes_)
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'gesture_model_{timestamp}.h5'
    model.save(model_name)
    print(f"\nModel saved as: {model_name}")
    
    # Save label encoder classes
    np.save(f'label_classes_{timestamp}.npy', label_encoder.classes_)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{timestamp}.png')
    plt.show()
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{timestamp}.png')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, 
                                target_names=label_encoder.classes_))

if __name__ == "__main__":
    main()