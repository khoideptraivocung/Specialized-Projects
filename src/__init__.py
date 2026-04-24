"""
Khởi tạo module src - Chuẩn bị phân tích dữ liệu
"""

import os
from pathlib import Path

# ============ Cấu hình đường dẫn ============
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR/"data_plotting"
# ============ Danh sách người phụ trách ============
PEOPLE = ["chi", "huy", "khanh", "khoi", "ninh", "phhuy","1", "2", "3","4"]

# ============ Danh sách file dữ liệu ============
DATA_FILES = [2, 4, 6, 8, 10]

# ============ Cấu hình pandas ============
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# ============ Logging ============
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ============ Hàm tiện ích ============
def get_data_path(person: str, file_num: int) -> Path:
    """Lấy đường dẫn file dữ liệu"""
    return DATA_DIR / person / f"{file_num}.csv"

def validate_data_exists() -> bool:
    """Kiểm tra tất cả file dữ liệu có tồn tại không"""
    for person in PEOPLE:
        for file_num in DATA_FILES:
            path = get_data_path(person, file_num)
            if not path.exists():
                logger.warning(f"Thiếu file: {path}")
                return False
    logger.info("✓ Tất cả file dữ liệu đều tồn tại")
    return True

__all__ = ['BASE_DIR', 'DATA_DIR', 'PEOPLE', 'DATA_FILES', 'logger', 'get_data_path', 'validate_data_exists']