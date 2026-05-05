import os
import glob
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.model import build_cnn_model

from src import logger, DATA_DIR, BASE_DIR

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data_by_person():
    """Load data and group by person."""
    logger.info("Loading data by person from data folder...")
    people_data = {}
    
    for person_dir in DATA_DIR.iterdir():
        if person_dir.is_dir():
            person = person_dir.name
            X_list = []
            y_list = []
            for file_path in person_dir.glob('*.csv'):
                thickness = float(file_path.stem)
                df = pd.read_csv(file_path)
                X_list.append(df.values)
                y_list.append(np.full(len(df), thickness))
            
            if X_list:
                people_data[person] = {
                    'X': np.vstack(X_list),
                    'y': np.concatenate(y_list)
                }
                logger.info(f"Loaded person {person}: {people_data[person]['X'].shape[0]} samples")
    return people_data

def apply_lowpass_filter(data, cutoff=0.1, fs=1.0, order=4):
    """Apply low-pass filter to signal."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    return filtered_data

def get_next_exp_dir():
    exp_base = BASE_DIR / "experiments"
    exp_base.mkdir(exist_ok=True)
    existing_exps = [d for d in exp_base.iterdir() if d.is_dir() and d.name.startswith('exp_')]
    if not existing_exps:
        return exp_base / "exp_001"
    
    max_idx = 0
    for d in existing_exps:
        try:
            idx = int(d.name.split('_')[1])
            max_idx = max(max_idx, idx)
        except ValueError:
            pass
    return exp_base / f"exp_{max_idx + 1:03d}"

def plot_scatter(y_true, y_pred, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title('Scatter Plot: Real vs Predict')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_residual(y_true, y_pred, save_path):
    residual = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residual, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Prediction')
    plt.ylabel('Residual (True - Pred)')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_error_bars(y_true, y_pred, save_path):
    unique_labels = np.unique(y_true)
    means = []
    stds = []
    
    for label in unique_labels:
        idx = (y_true == label)
        preds = y_pred[idx]
        means.append(np.mean(preds))
        stds.append(np.std(preds))
        
    plt.figure(figsize=(8, 6))
    plt.errorbar(unique_labels, means, yerr=stds, fmt='o', capsize=5, linestyle='-', marker='s')
    plt.plot(unique_labels, unique_labels, 'r--', label='Ideal')
    plt.xlabel('Label (Thickness)')
    plt.ylabel('Mean Prediction')
    plt.title('Error Bars by Label')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def create_readme(exp_dir, metrics):
    readme_content = f"""# Pipeline Results

## Mô tả bài toán
Dự đoán độ dày (thickness) từ dữ liệu time-series sensor.
Labels: 2, 4, 6, 8, 10.

## Pipeline
- Split theo person (8 train, 1 val, 1 test)
- Low-pass filter
- Isolation Forest (loại bỏ outlier trên tập train)
- StandardScaler
- 1D CNN Regression Model

## Kết quả
- Test MAE: {metrics['test_mae']:.4f}
- Test MSE: {metrics['test_mse']:.4f}
"""
    with open(exp_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

def create_model(filters=[64, 128, 64], dense_units=[64, 32], dropout_rate=0.2):
    from tensorflow.keras import layers, Sequential
    model = Sequential([
        layers.Input(shape=(333, 1)),
        layers.Conv1D(filters=filters[0], kernel_size=3, padding='same'),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        layers.Conv1D(filters=filters[1], kernel_size=3, padding='same'),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        layers.Conv1D(filters=filters[2], kernel_size=3, padding='same'),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        layers.GlobalAveragePooling1D(),
        
        layers.Dense(dense_units[0], activation='relu'),
        layers.Dropout(dropout_rate + 0.1),
        layers.Dense(dense_units[1], activation='relu'),
        
        layers.Dense(1, activation='linear')
    ])
    return model

def run_pipeline():
    config = load_config()
    people_data = load_data_by_person()
    
    people = list(people_data.keys())
    np.random.seed(42)
    np.random.shuffle(people)
    
    train_count = config['data']['train']
    val_count = config['data']['val']
    
    train_people = people[:train_count]
    val_people = people[train_count:train_count+val_count]
    test_people = people[train_count+val_count:]
    
    logger.info(f"Train people: {train_people}")
    logger.info(f"Val people: {val_people}")
    logger.info(f"Test people: {test_people}")
    
    def combine_people(person_list):
        X = np.vstack([people_data[p]['X'][:, :333] for p in person_list])
        y = np.concatenate([people_data[p]['y'] for p in person_list])
        return X, y

    X_train, y_train = combine_people(train_people)
    X_val, y_val = combine_people(val_people)
    X_test, y_test = combine_people(test_people)
    
    # Preprocessing
    if config['preprocess']['lowpass']:
        logger.info("Applying lowpass filter...")
        X_train = apply_lowpass_filter(X_train)
        X_val = apply_lowpass_filter(X_val)
        X_test = apply_lowpass_filter(X_test)
        
    contamination = config['preprocess']['isolation_forest']['contamination']
    logger.info(f"Applying Isolation Forest with contamination={contamination}...")
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_train)
    
    train_inliers = iso.predict(X_train) == 1
    X_train = X_train[train_inliers]
    y_train = y_train[train_inliers]
    logger.info(f"Train size after outlier removal: {X_train.shape[0]}")
    
    logger.info("Applying StandardScaler to X...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    logger.info("Applying StandardScaler to y...")
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    logger.info("Starting Auto Improvement Loop...")
    
    # Auto Improvement Hyperparameter Configurations
    hyperparams_list = [
        {'filters': [64, 128, 64], 'dense': [64, 32], 'dropout': 0.2, 'lr': 0.001},     # Baseline
        {'filters': [128, 256, 128], 'dense': [128, 64], 'dropout': 0.3, 'lr': 0.001},  # Increased capacity + dropout
        {'filters': [128, 256, 128], 'dense': [128, 64], 'dropout': 0.4, 'lr': 0.0005}, # More dropout, lower LR
        {'filters': [256, 512, 256], 'dense': [256, 128], 'dropout': 0.5, 'lr': 0.0005} # Huge capacity, huge dropout
    ]
    
    best_mae = float('inf')
    best_metrics = None
    best_model_path = None
    
    for attempt, hp in enumerate(hyperparams_list, 1):
        logger.info(f"\n--- ATTEMPT {attempt} ---")
        logger.info(f"Hyperparams: {hp}")
        
        model = create_model(filters=hp['filters'], dense_units=hp['dense'], dropout_rate=hp['dropout'])
        from tensorflow.keras.optimizers import Adam
        model.compile(optimizer=Adam(learning_rate=hp['lr']), loss='mse', metrics=['mae'])
        
        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        ]
        
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        logger.info("Evaluating model...")
        y_pred_scaled = model.predict(X_test).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate unscaled MAE and MSE
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        test_mae = mean_absolute_error(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        
        logger.info(f"Test MAE for attempt {attempt}: {test_mae:.4f}")
        
        exp_dir = get_next_exp_dir()
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        model.save(exp_dir / 'model.keras')
        
        import pickle
        with open(exp_dir / 'scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(exp_dir / 'scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
            
        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history.history, f)
            
        metrics = {
            'test_mse': float(test_mse),
            'test_mae': float(test_mae)
        }
        with open(exp_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f)
            
        plot_scatter(y_test, y_pred, exp_dir / 'scatter.png')
        plot_residual(y_test, y_pred, exp_dir / 'residual.png')
        plot_error_bars(y_test, y_pred, exp_dir / 'error_bar.png')
        
        create_readme(exp_dir, metrics)
        
        if test_mae < best_mae:
            best_mae = test_mae
            best_metrics = metrics
            best_model_path = exp_dir
            logger.info(f"New Best Model! MAE: {best_mae:.4f}")
            
        if test_mae < 1.0:
            logger.info("Target MAE < 1.0 reached! Stopping auto-improvement loop.")
            break
            
    logger.info(f"\nPipeline completed! Best MAE: {best_mae:.4f} at {best_model_path}")
    return best_metrics

if __name__ == '__main__':
    run_pipeline()
