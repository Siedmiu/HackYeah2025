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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ==================== CONFIGURATION ====================
WINDOW_SIZE = 35  # Number of samples per window
WINDOW_STEP = 25  # Overlap between windows (50% overlap)
AUGMENTATION_FACTOR = 20  # Augmented versions per original window

# Gesture to sensor mapping
GESTURE_SENSORS = {
    'Jumping': ['adxl_ax', 'adxl_ay', 'adxl_az'],
    # 'Kucania': ['mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz'],
    'Udarzenia': ['l3gd_gx', 'l3gd_gy', 'l3gd_gz', 'lsm_ax', 'lsm_ay', 'lsm_az'],
    # 'Udarzenia_lewa': ['mpu_ax', 'mpu_ay', 'mpu_az', 'mpu_gx', 'mpu_gy', 'mpu_gz'],
    # 'Strzal': ['adxl_ax', 'adxl_ay', 'adxl_az', 'lsm_ax', 'lsm_ay', 'lsm_az'],
    'Syf': ['mpu_ax', 'mpu_ay', 'mpu_az', 'adxl_ax', 'adxl_ay', 'adxl_az']
}

# ==================== DATA AUGMENTATION ====================
def add_noise(data, noise_level=0.05):
    """Add Gaussian noise"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise * np.std(data, axis=0)

def scale_data(data, scale_range=(0.9, 1.1)):
    """Random scaling"""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale

def time_warp(data, sigma=0.2):
    """Time warping"""
    num_samples = data.shape[0]
    warp = np.random.normal(1.0, sigma, num_samples)
    warp = np.cumsum(warp)
    warp = (warp - warp.min()) / (warp.max() - warp.min()) * (num_samples - 1)
    indices = np.clip(warp, 0, num_samples - 1).astype(int)
    return data[indices]

def augment_window(window, num_augmentations=5):
    """Apply augmentations to create variations"""
    augmented = []
    
    for _ in range(num_augmentations):
        aug = window.copy()
        
        # Apply noise
        aug = add_noise(aug, noise_level=np.random.uniform(0.03, 0.12))
        
        # Apply scaling
        aug = scale_data(aug, scale_range=(0.8, 1.2))
        
        # Apply time warping (70% chance)
        if np.random.random() > 0.3:
            aug = time_warp(aug, sigma=np.random.uniform(0.15, 0.35))
        
        # Small rotation for 3-axis data
        if np.random.random() > 0.5 and aug.shape[1] >= 3:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for i in range(0, min(aug.shape[1] - 1, 2)):
                temp = aug[:, i].copy()
                aug[:, i] = cos_a * temp - sin_a * aug[:, i+1]
                aug[:, i+1] = sin_a * temp + cos_a * aug[:, i+1]
        
        augmented.append(aug)
    
    return augmented

# ==================== DATA LOADING ====================
def load_merged_data(data_dir='dataset/merged'):
    """Load merged CSV files"""
    all_windows = []
    all_labels = []
    max_features = 0  # Track maximum number of features
    
    csv_files = glob.glob(os.path.join(data_dir, '*_merged.csv'))
    
    if not csv_files:
        raise ValueError(f"No merged CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} merged files\n")
    
    # First pass: determine max feature count
    temp_windows = []
    temp_labels = []
    
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        
        if 'Activity_Type' not in df.columns or len(df) == 0:
            continue
        
        gesture = df['Activity_Type'].iloc[0]
        sensor_cols = GESTURE_SENSORS.get(gesture)
        
        if not sensor_cols:
            continue
        
        available_sensors = [col for col in sensor_cols if col in df.columns]
        if not available_sensors:
            continue
        
        max_features = max(max_features, len(available_sensors))
        
        if 'Timestamp' in df.columns:
            df = df.sort_values('Timestamp')
        
        sensor_data = df[available_sensors].values
        
        num_windows = 0
        for i in range(0, len(sensor_data) - WINDOW_SIZE + 1, WINDOW_STEP):
            window = sensor_data[i:i + WINDOW_SIZE]
            if window.shape[0] == WINDOW_SIZE:
                temp_windows.append(window)
                temp_labels.append(gesture)
                num_windows += 1
        
        print(f"{os.path.basename(csv_file):<50} {len(df):>4} rows → {num_windows:>3} windows ({gesture})")
    
    # Second pass: pad all windows to max_features
    print(f"\nPadding all windows to {max_features} features...")
    for window in temp_windows:
        if window.shape[1] < max_features:
            # Pad with zeros
            padding = np.zeros((WINDOW_SIZE, max_features - window.shape[1]))
            padded_window = np.concatenate([window, padding], axis=1)
            all_windows.append(padded_window)
        else:
            all_windows.append(window)
    
    all_labels = temp_labels
    
    if not all_windows:
        raise ValueError("No windows created! Check your data files.")
    
    return np.array(all_windows), np.array(all_labels)

# ==================== MODEL ARCHITECTURE ====================
def create_cnn_lstm_model(input_shape, num_classes):
    """CNN-LSTM hybrid model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # CNN feature extraction
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # LSTM temporal patterns
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

# ==================== MAIN PIPELINE ====================
def main():
    print("=" * 70)
    print("GESTURE RECOGNITION - MERGED FILES TRAINING")
    print("=" * 70)
    
    # 1. Load data and create windows
    print("\n[1/7] Loading merged files and creating windows...")
    windows, labels = load_merged_data('dataset/merged')
    
    print(f"\nTotal windows created: {len(windows)}")
    print(f"Window shape: {windows[0].shape}")
    print(f"\nGesture distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for gesture, count in zip(unique, counts):
        print(f"  {gesture:<20} {count:>4} windows")
    
    # 2. Shuffle data
    print("\n[2/7] Shuffling windows...")
    shuffle_idx = np.random.permutation(len(windows))
    windows = windows[shuffle_idx]
    labels = labels[shuffle_idx]
    
    # 3. Split BEFORE augmentation
    print("\n[3/7] Splitting into train/test sets...")
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_idx, test_idx in sss.split(windows, labels):
        windows_train, windows_test = windows[train_idx], windows[test_idx]
        labels_train, labels_test = labels[train_idx], labels[test_idx]
    
    print(f"Train set: {len(windows_train)} windows")
    print(f"Test set: {len(windows_test)} windows")
    
    # 4. Augment training data only
    print("\n[4/7] Augmenting training data...")
    augmented_windows = []
    augmented_labels = []
    
    for window, label in zip(windows_train, labels_train):
        # Add original
        augmented_windows.append(window)
        augmented_labels.append(label)
        
        # Add augmented versions
        aug_windows = augment_window(window, AUGMENTATION_FACTOR)
        augmented_windows.extend(aug_windows)
        augmented_labels.extend([label] * len(aug_windows))
    
    X_train = np.array(augmented_windows)
    y_train = np.array(augmented_labels)
    X_test = windows_test
    y_test = labels_test
    
    print(f"After augmentation - Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 5. Encode labels
    print("\n[5/7] Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    y_train_categorical = keras.utils.to_categorical(y_train_encoded)
    y_test_categorical = keras.utils.to_categorical(y_test_encoded)
    
    print(f"Classes: {label_encoder.classes_}")
    
    # 6. Normalize
    print("\n[6/7] Normalizing data...")
    n_train, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_train, n_timesteps, n_features)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(len(X_test), n_timesteps, n_features)
    
    # 7. Build and train
    print("\n[7/7] Building and training model...")
    
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
        patience=20,
        restore_best_weights=True,
        min_delta=0.001
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train
    history = model.fit(
        X_train_scaled, y_train_categorical,
        validation_data=(X_test_scaled, y_test_categorical),
        epochs=100,
        batch_size=min(32, len(X_train_scaled) // 4),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_categorical, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f'gesture_model_{timestamp}.keras')
    np.save(f'label_classes_{timestamp}.npy', label_encoder.classes_)
    print(f"\nModel saved as: gesture_model_{timestamp}.keras")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{timestamp}.png', dpi=150)
    print(f"Training plot saved: training_history_{timestamp}.png")
    plt.close()
    
    # Confusion matrix
    y_pred = model.predict(X_test_scaled, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_categorical, axis=1)
    
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{timestamp}.png', dpi=150)
    print(f"Confusion matrix saved: confusion_matrix_{timestamp}.png")
    plt.close()
    
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, 
                                target_names=label_encoder.classes_))

if __name__ == "__main__":
    main()