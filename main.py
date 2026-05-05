import sys
from src.data_loader import (lay_du_lieu, ve_du_lieu_theo_nguoi, 
                             ve_du_lieu_lowpass_filter, tinh_mean_vector_va_cosine_similarity,
                             compare_people_per_label, save_filtered_data, ve_du_lieu_processed)

def main():
    print("--- Bắt đầu chương trình ---")
    
    # Lưu dữ liệu sau lowpass filter
    
    
    # Vẽ dữ liệu đã lưu
    ve_du_lieu_processed()

if __name__ == "__main__":
    main()