"""
1D CNN Regression Model for Thickness Prediction
Input: (N, 333, 1) - 333 time-series features
Output: Continuous value (thickness)
"""

from tensorflow import keras
from tensorflow.keras import layers, Sequential
from src import logger


def build_cnn_model(input_shape: tuple = (333, 1)) -> Sequential:
    """
    Build 1D CNN model for regression
    
    Args:
        input_shape: Input shape (features, 1) - default (333, 1)
    
    Returns:
        Compiled Keras model
    """
    logger.info("=" * 70)
    logger.info("BUILDING 1D CNN REGRESSION MODEL")
    logger.info("=" * 70)
    
    model = Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Conv block
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same',
            name='conv1d_1'
        ),
        layers.ReLU(name='relu_1'),
        layers.MaxPooling1D(pool_size=2, name='maxpool_1'),
        layers.Dropout(0.2, name='dropout_1'),
        
        # Second Conv block
        layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            name='conv1d_2'
        ),
        layers.ReLU(name='relu_2'),
        layers.MaxPooling1D(pool_size=2, name='maxpool_2'),
        layers.Dropout(0.2, name='dropout_2'),
        
        # Third Conv block
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same',
            name='conv1d_3'
        ),
        layers.ReLU(name='relu_3'),
        layers.MaxPooling1D(pool_size=2, name='maxpool_3'),
        layers.Dropout(0.2, name='dropout_3'),
        
        # Global pooling - flattens the sequence
        layers.GlobalAveragePooling1D(name='global_avgpool'),
        
        # Dense layers for regression
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.3, name='dropout_dense'),
        layers.Dense(32, activation='relu', name='dense_2'),
        
        # Output layer - single value for regression
        layers.Dense(1, activation='linear', name='output')
    ], name='cnn_regression_model')
    
    logger.info("\n✓ Model architecture created:")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Output: Single value (regression)")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    logger.info("\n✓ Model compiled:")
    logger.info(f"  Optimizer: adam")
    logger.info(f"  Loss: mse")
    logger.info(f"  Metrics: mae")
    logger.info("=" * 70 + "\n")
    
    return model


def print_model_summary(model: Sequential) -> None:
    """
    Print detailed model summary
    
    Args:
        model: Keras model
    """
    logger.info("=" * 70)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 70)
    model.summary(print_fn=lambda x: logger.info(x))
    logger.info("=" * 70 + "\n")
