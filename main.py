import sys
from src.data_loader import (lay_du_lieu, tinh_tong, ve_du_lieu_theo_nguoi, 
                             ve_du_lieu_lowpass_filter, tinh_mean_vector_va_cosine_similarity,
                             compare_people_per_label)

def main():
    print("--- Bắt đầu chương trình ---")
    
    # So sánh các người trong mỗi nhãn (2, 4, 6, 8, 10)
    compare_people_per_label()

if __name__ == "__main__":
    main()