import numpy as np
from src import PEOPLE, DATA_FILES, get_data_path, logger, PROCESS_DIR
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def get_processed_data_path(person: str, file_num: int) -> Path:
    """Lấy đường dẫn file dữ liệu processed"""
    return PROCESS_DIR / person / f"{file_num}.csv"


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize dữ liệu bằng Z-score normalization
    (x - mean) / std
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Tránh chia cho 0
    std[std == 0] = 1
    return (data - mean) / std

def gop_du_lieu_va_kiem_tra():
    """
    Gộp tất cả file (2, 4, 6, 8, 10) từ tất cả người thành 1 array
    Đọc từ PROCESSED DATA
    Kiểm tra NaN và shape
    Return: numpy array đã gộp
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU GỘP DỮ LIỆU (từ PROCESSED)")
    logger.info("=" * 60)
    
    try:
        inputs = []
        
        logger.info(f"\n📂 ĐANG ĐỌC TẤT CẢ FILE PROCESSED:")
        
        # Đọc tất cả file từ tất cả người (từ processed folder)
        for file_num in DATA_FILES:
            for person in PEOPLE:
                file_path = get_processed_data_path(person, file_num)
                df = pd.read_csv(file_path)
                inputs.append(df.values)  # Lấy raw array
                logger.info(f"   ✓ {person} - File {file_num}: shape {df.shape}")
        
        # Gộp tất cả array thành 1
        merged_array = np.vstack(inputs)  # Stack tất cả
        logger.info(f"\n   → Shape sau vstack: {merged_array.shape}")
        
        # ===== Kiểm tra =====
        logger.info(f"\n📊 KẾT QUẢ CUỐI CÙNG:")
        logger.info(f"   Shape: {merged_array.shape}")
        
        # Kiểm tra NaN
        nan_count = np.isnan(merged_array).sum()
        if nan_count > 0:
            logger.warning(f"   ⚠️  NaN found: {nan_count} NaN values")
        else:
            logger.info(f"   ✓ Không có dữ liệu NaN")
        
        logger.info(f"   Columns: {merged_array.shape[1]}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ HOÀN THÀNH GỘP DỮ LIỆU")
        logger.info("=" * 60)
        
        return merged_array
    
    except Exception as e:
        logger.error(f"✗ Lỗi: {e}")
        return None


def tao_y_voi_noise(X: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """
    Tạo y từ file number (2, 4, 6, 8, 10) với noise nhẹ
    Đọc từ PROCESSED DATA
    
    Args:
        X: Dữ liệu đã merge, shape (n_samples, n_features)
        noise_level: Mức độ noise (%)
    
    Returns:
        y: Array label cho mỗi sample
    """
    logger.info("=" * 60)
    logger.info("TẠO Y VỚI NOISE NHẸ (từ PROCESSED)")
    logger.info("=" * 60)
    
    y = []
    
    logger.info(f"\n📂 ĐANG GẮN Y CHO TỪNG SAMPLE:")
    
    # Lặp lại quy trình gộp dữ liệu để lấy y value (từ processed data)
    for file_num in DATA_FILES:
        for person in PEOPLE:
            file_path = get_processed_data_path(person, file_num)
            df = pd.read_csv(file_path)
            
            # Mỗi sample từ file này sẽ có y = file_num
            n_samples = len(df)
            y_values = np.full(n_samples, float(file_num))
            
            y.extend(y_values)
            logger.info(f"   ✓ {person} - File {file_num}: {n_samples} samples → y = {file_num}")
    
    y = np.array(y, dtype=np.float32)
    
    logger.info(f"\n   → Shape y trước noise: {y.shape}")
    
    # Thêm noise nhẹ
    noise = np.random.normal(0, noise_level * np.mean(y), y.shape)
    y_with_noise = y + noise
    
    logger.info(f"   ✓ Thêm noise với mức độ: {noise_level*100:.1f}%")
    logger.info(f"   → Shape y sau noise: {y_with_noise.shape}")
    logger.info(f"   → Y min: {y_with_noise.min():.4f}, max: {y_with_noise.max():.4f}")
    logger.info(f"   → Y mean: {y_with_noise.mean():.4f}, std: {y_with_noise.std():.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ HOÀN THÀNH TẠO Y")
    logger.info("=" * 60)
    
    return y_with_noise


def scale_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Áp dụng StandardScaler cho X và y
    
    Args:
        X: Features, shape (n_samples, n_features)
        y: Labels, shape (n_samples,)
    
    Returns:
        X_scaled, y_scaled, scaler_X, scaler_y
    """
    logger.info("=" * 60)
    logger.info("STANDARDSCALER XỬ LÝ")
    logger.info("=" * 60)
    
    # Reshape y cho scaler (mong đợi 2D array)
    y_reshaped = y.reshape(-1, 1)
    
    # Tạo scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit và transform
    logger.info(f"\n📊 TRƯỚC SCALING:")
    logger.info(f"   X - shape: {X.shape}, mean: {X.mean():.4f}, std: {X.std():.4f}")
    logger.info(f"   y - mean: {y.mean():.4f}, std: {y.std():.4f}")
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_reshaped).ravel()
    
    logger.info(f"\n📊 SAU SCALING:")
    logger.info(f"   X - shape: {X_scaled.shape}, mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")
    logger.info(f"   y - mean: {y_scaled.mean():.4f}, std: {y_scaled.std():.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ HOÀN THÀNH STANDARDSCALER")
    logger.info("=" * 60)
    
    return X_scaled, y_scaled, scaler_X, scaler_y


def load_and_preprocess(noise_level: float = 0.05) -> dict:
    """
    Pipeline hoàn chỉnh: Load → Tạo y → StandardScaler
    
    Args:
        noise_level: Mức độ noise cho y
    
    Returns:
        dict với keys: X, y, X_scaled, y_scaled, scaler_X, scaler_y
    """
    logger.info("\n\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 58 + "║")
    logger.info("║" + "  PIPELINE: LOAD → TẠO Y → STANDARDSCALER".center(58) + "║")
    logger.info("║" + " " * 58 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    
    # Bước 1: Load data
    X = gop_du_lieu_va_kiem_tra()
    if X is None:
        logger.error("✗ Không thể load dữ liệu")
        return None
    
    # Bước 2: Tạo y với noise
    y = tao_y_voi_noise(X, noise_level=noise_level)
    if y is None:
        logger.error("✗ Không thể tạo y")
        return None
    
    # Bước 3: Áp dụng StandardScaler
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data(X, y)
    
    logger.info("\n" + "=" * 60)
    logger.info("TỔNG HỢP DỮ LIỆU CUỐI CÙNG")
    logger.info("=" * 60)
    logger.info(f"Original X shape: {X.shape}")
    logger.info(f"Scaled X shape: {X_scaled.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"y_scaled shape: {y_scaled.shape}")
    logger.info("=" * 60 + "\n")
    
    return {
        'X': X.astype(np.float32),
        'y': y.astype(np.float32),
        'X_scaled': X_scaled.astype(np.float32),
        'y_scaled': y_scaled.astype(np.float32),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

if __name__ == "__main__":
    # Chạy pipeline hoàn chỉnh
    result = load_and_preprocess(noise_level=0.05)
    
    if result:
        logger.info("\n✅ Pipeline hoàn tất thành công!")
        logger.info(f"   X_scaled: {result['X_scaled'].shape}")
        logger.info(f"   y_scaled: {result['y_scaled'].shape}")
