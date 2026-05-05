"""
Training pipeline for 1D CNN Regression Model
Loads processed data, trains model, and saves it
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

from src import logger, PROCESS_DIR, RESULT_DIR
from src.preprocess import load_and_preprocess
from src.model import build_cnn_model, print_model_summary


def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """
    Reshape X from (N, 333) to (N, 333, 1) for 1D CNN
    
    Args:
        X: Original features, shape (N, 333)
    
    Returns:
        Reshaped features, shape (N, 333, 1)
    """
    logger.info("=" * 70)
    logger.info("RESHAPING DATA FOR 1D CNN")
    logger.info("=" * 70)
    
    logger.info(f"\n   Original shape: {X.shape}")
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
    logger.info(f"   Reshaped to: {X_reshaped.shape}")
    
    logger.info("=" * 70 + "\n")
    return X_reshaped


def split_train_val_test(X: np.ndarray, y: np.ndarray, 
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1,
                          test_ratio: float = 0.1,
                          random_state: int = 42) -> tuple:
    """
    Split data into train/val/test (80/10/10)
    
    Args:
        X: Features
        y: Labels
        train_ratio: Training set ratio (default 80%)
        val_ratio: Validation set ratio (default 10%)
        test_ratio: Test set ratio (default 10%)
        random_state: Random seed
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("=" * 70)
    logger.info("SPLITTING DATA (80 / 10 / 10)")
    logger.info("=" * 70)
    
    total_samples = len(X)
    logger.info(f"\n   Total samples: {total_samples}")
    
    # First split: train + val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_state
    )
    
    # Second split: train / val from remaining
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state
    )
    
    logger.info(f"\n   Train set: {len(X_train)} samples ({len(X_train)/total_samples*100:.1f}%)")
    logger.info(f"   Val set:   {len(X_val)} samples ({len(X_val)/total_samples*100:.1f}%)")
    logger.info(f"   Test set:  {len(X_test)} samples ({len(X_test)/total_samples*100:.1f}%)")
    
    logger.info(f"\n   X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"   X_val shape:   {X_val.shape}, y_val shape: {y_val.shape}")
    logger.info(f"   X_test shape:  {X_test.shape}, y_test shape: {y_test.shape}")
    
    logger.info("=" * 70 + "\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model, 
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100,
                batch_size: int = 32,
                patience: int = 10) -> dict:
    """
    Train the model with EarlyStopping
    
    Args:
        model: Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Max epochs
        batch_size: Batch size
        patience: EarlyStopping patience
    
    Returns:
        Training history dict
    """
    logger.info("=" * 70)
    logger.info("TRAINING MODEL")
    logger.info("=" * 70)
    
    logger.info(f"\n   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   EarlyStopping patience: {patience}")
    
    # EarlyStopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ TRAINING COMPLETED")
    logger.info("=" * 70 + "\n")
    
    return history


def evaluate_model(model, 
                   X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model on test set
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Evaluation metrics dict
    """
    logger.info("=" * 70)
    logger.info("EVALUATING MODEL ON TEST SET")
    logger.info("=" * 70)
    
    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    logger.info(f"\n   Test Loss (MSE): {test_loss:.6f}")
    logger.info(f"   Test MAE:        {test_mae:.6f}")
    
    logger.info("\n" + "=" * 70 + "\n")
    
    return {
        'test_loss': test_loss,
        'test_mae': test_mae
    }


def save_model(model, model_path: str = None) -> None:
    """
    Save model in .keras format
    
    Args:
        model: Trained Keras model
        model_path: Path to save model (default: processed/model.keras)
    """
    if model_path is None:
        model_path = PROCESS_DIR / "model.keras"
    else:
        model_path = Path(model_path)
    
    # Create directory if not exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("SAVING MODEL")
    logger.info("=" * 70)
    
    model.save(str(model_path))
    
    logger.info(f"\n   ✓ Model saved to: {model_path}")
    logger.info(f"   File size: {model_path.stat().st_size / 1024:.2f} KB")
    
    logger.info("=" * 70 + "\n")


def plot_training_history(history, save_dir: str = None) -> None:
    """
    Plot training history (loss and mae) and save as images
    
    Args:
        history: History object from model.fit()
        save_dir: Directory to save plots (default: result/)
    """
    if save_dir is None:
        save_dir = RESULT_DIR
    else:
        save_dir = Path(save_dir)
    
    # Create directory if not exists
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("PLOTTING TRAINING HISTORY")
    logger.info("=" * 70)
    
    # Extract history data
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs_range = range(1, len(loss) + 1)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    axes[0].plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Loss - Training vs Validation', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: MAE
    axes[1].plot(epochs_range, mae, 'g-', label='Training MAE', linewidth=2)
    axes[1].plot(epochs_range, val_mae, 'orange', label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('MAE', fontsize=12, fontweight='bold')
    axes[1].set_title('Model MAE - Training vs Validation', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_path = save_dir / "training_history_combined.png"
    plt.savefig(str(combined_path), dpi=300, bbox_inches='tight')
    logger.info(f"\n   ✓ Combined plot saved: {combined_path}")
    
    # Save individual plots
    fig_loss = plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=4)
    plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=13, fontweight='bold')
    plt.title('Training Loss History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    loss_path = save_dir / "loss.png"
    plt.savefig(str(loss_path), dpi=300, bbox_inches='tight')
    logger.info(f"   ✓ Loss plot saved: {loss_path}")
    plt.close(fig_loss)
    
    # MAE plot
    fig_mae = plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, mae, 'g-', label='Training MAE', linewidth=2.5, marker='o', markersize=4)
    plt.plot(epochs_range, val_mae, 'orange', label='Validation MAE', linewidth=2.5, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('MAE', fontsize=13, fontweight='bold')
    plt.title('Training MAE History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    mae_path = save_dir / "mae.png"
    plt.savefig(str(mae_path), dpi=300, bbox_inches='tight')
    logger.info(f"   ✓ MAE plot saved: {mae_path}")
    plt.close(fig_mae)
    
    plt.close('all')
    
    logger.info(f"\n   📊 All plots saved to: {save_dir}")
    logger.info("=" * 70 + "\n")


def train_full_pipeline(noise_level: float = 0.05,
                        epochs: int = 100,
                        batch_size: int = 32,
                        patience: int = 10,
                        model_save_path: str = None) -> dict:
    """
    Full training pipeline:
    1. Load & preprocess data
    2. Reshape for CNN
    3. Split train/val/test
    4. Build model
    5. Train with EarlyStopping
    6. Evaluate
    7. Plot training history
    8. Save model
    
    Args:
        noise_level: Noise level for y creation (default 0.05)
        epochs: Max training epochs (default 100)
        batch_size: Batch size (default 32)
        patience: EarlyStopping patience (default 10)
        model_save_path: Path to save model (default: processed/model.keras)
    
    Returns:
        Results dict with metrics and paths
    """
    logger.info("\n\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 68 + "║")
    logger.info("║" + "  FULL TRAINING PIPELINE".center(68) + "║")
    logger.info("║" + "  Load → Reshape → Split → Train → Evaluate → Plot → Save".center(68) + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    
    # Step 1: Load & preprocess
    logger.info("\n[1/8] LOADING DATA\n")
    data = load_and_preprocess(noise_level=noise_level)
    X = data['X_scaled']
    y = data['y']
    
    # Step 2: Reshape
    logger.info("\n[2/8] RESHAPING DATA\n")
    X = reshape_for_cnn(X)
    
    # Step 3: Split data
    logger.info("\n[3/8] SPLITTING DATA\n")
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Step 4: Build model
    logger.info("\n[4/8] BUILDING MODEL\n")
    model = build_cnn_model(input_shape=(333, 1))
    print_model_summary(model)
    
    # Step 5: Train
    logger.info("\n[5/8] TRAINING MODEL\n")
    history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience
    )
    
    # Step 6: Evaluate
    logger.info("\n[6/8] EVALUATING MODEL\n")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 6.5: Plot training history
    logger.info("\n[7/8] PLOTTING TRAINING HISTORY\n")
    plot_training_history(history, save_dir=RESULT_DIR)
    
    # Step 7: Save
    logger.info("\n[8/8] SAVING MODEL\n")
    save_model(model, model_save_path)
    
    # Summary
    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"\n   ✓ Test MAE (Mean Absolute Error): {metrics['test_mae']:.6f}")
    logger.info(f"   ✓ Test Loss (MSE):                 {metrics['test_loss']:.6f}")
    logger.info(f"\n   📊 Plots saved to: {RESULT_DIR}")
    logger.info(f"      - training_history_combined.png")
    logger.info(f"      - loss.png")
    logger.info(f"      - mae.png")
    logger.info(f"\n   Data shapes:")
    logger.info(f"     - X_train: {X_train.shape}")
    logger.info(f"     - X_val:   {X_val.shape}")
    logger.info(f"     - X_test:  {X_test.shape}")
    logger.info(f"\n   Model parameters: {model.count_params():,}")
    logger.info("=" * 70 + "\n")
    
    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'model_path': model_save_path or (PROCESS_DIR / "model.keras")
    }


if __name__ == "__main__":
    # Run full training pipeline
    results = train_full_pipeline(
        noise_level=0.05,
        epochs=100,
        batch_size=32,
        patience=10,
        model_save_path=None  # Will use default: processed/model.keras
    )
    
    logger.info("\n✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"   Final Test MAE: {results['metrics']['test_mae']:.6f}")
