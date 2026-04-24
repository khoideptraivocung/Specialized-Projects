"""
Module lấy và xử lý dữ liệu
"""

import numpy as np
from src import PEOPLE, DATA_FILES, get_data_path, logger

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

def ve_du_lieu_theo_nguoi():
    """
    2 cột, 5 hàng (mỗi cột 5 người), mỗi file 1 màu (raw data)
    """

    logger.info("Bắt đầu vẽ dữ liệu (raw)...")

    n_people = len(PEOPLE)
    fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)

    # Màu sắc bình thường
    colors = plt.cm.tab10.colors
    
    # 🎨 Thiết lập style đẹp
    plt.style.use('seaborn-v0_8-darkgrid')
    fig.patch.set_facecolor('white')

    try:
        for i, person in enumerate(PEOPLE):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            ax.set_facecolor('#F8F9FA')

            for j, file_num in enumerate(DATA_FILES):
                path = get_data_path(person, file_num)
                df = pd.read_csv(path)

                # 🔥 lấy 1 sample (raw)
                sample = df.iloc[0].values.astype(float)

                ax.plot(sample, color=colors[j], 
                       label=f'Thickness {file_num}', linewidth=2.5, alpha=0.8)

            ax.set_title(f' {person.upper()}', fontsize=12, fontweight='bold', pad=10)
            ax.grid(alpha=0.25, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Cài đặt label cho trục
            ax.set_ylabel("Value", fontsize=10, fontweight='bold')

        # Legend chung ở dưới cùng
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                  ncol=5, fontsize=10, framealpha=0.95, edgecolor='gray', fancybox=True)

        # Cài đặt xlabel cho hàng cuối
        axes[4, 0].set_xlabel("Feature Index", fontsize=11, fontweight='bold')
        axes[4, 1].set_xlabel("Feature Index", fontsize=11, fontweight='bold')

        # Tiêu đề tổng quát đẹp hơn
        plt.suptitle("Raw Data Comparison Across All Samples", 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()

        output_path = get_data_path.__globals__['PLOT_DIR'] / 'compare_raw.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        plt.show()
        logger.info(f"✓ Đã lưu hình ảnh: {output_path}")

    except Exception as e:
        logger.error(f"Lỗi khi vẽ: {e}")

def ve_du_lieu_lowpass_filter(cutoff=0.1, order=4):
    """
    Vẽ dữ liệu sau khi áp dụng lowpass filter
    cutoff: tần số cắt normalization (0-1)
    order: độ của filter
    """

    logger.info(f"Bắt đầu vẽ dữ liệu với lowpass filter (cutoff={cutoff}, order={order})...")

    n_people = len(PEOPLE)
    fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)

    # Màu sắc bình thường
    colors = plt.cm.tab10.colors
    
    # 🎨 Thiết lập style đẹp
    plt.style.use('seaborn-v0_8-darkgrid')
    fig.patch.set_facecolor('white')

    # Thiết kế lowpass filter
    b, a = butter(order, cutoff, btype='low')

    try:
        for i, person in enumerate(PEOPLE):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            ax.set_facecolor('#F8F9FA')

            for j, file_num in enumerate(DATA_FILES):
                path = get_data_path(person, file_num)
                df = pd.read_csv(path)

                # 🔥 lấy 1 sample (raw)
                sample = df.iloc[0].values.astype(float)
                
                # Áp dụng lowpass filter
                filtered_sample = filtfilt(b, a, sample)

                ax.plot(filtered_sample, color=colors[j], 
                       label=f'Thickness {file_num}', linewidth=2.5, alpha=0.8)

            ax.set_title(f' {person.upper()}', fontsize=12, fontweight='bold', pad=10)
            ax.grid(alpha=0.25, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Cài đặt label cho trục
            ax.set_ylabel("Value", fontsize=10, fontweight='bold')

        # Cài đặt xlabel cho hàng cuối
        axes[4, 0].set_xlabel("Feature Index", fontsize=11, fontweight='bold')
        axes[4, 1].set_xlabel("Feature Index", fontsize=11, fontweight='bold')

        # Legend chung ở dưới cùng
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                  ncol=5, fontsize=10, framealpha=0.95, edgecolor='gray', fancybox=True)

        # Tiêu đề tổng quát
        plt.suptitle(" Lowpass Filtered Data Comparison", 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()

        output_path = get_data_path.__globals__['PLOT_DIR'] / 'compare_lowpass_filter.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

        plt.show()
        logger.info(f"Đã lưu hình ảnh: {output_path}")

    except Exception as e:
        logger.error(f"Lỗi khi vẽ: {e}")

def save_filtered_data(cutoff=0.1, order=4):
    """
    Lưu dữ liệu sau khi áp dụng lowpass filter vào folder 'processed'
    
    Cấu trúc folder:
    processed/
        chi/
            2.csv
            4.csv
            ...
        huy/
            2.csv
            ...
        ...
    
    cutoff: tần số cắt normalization (0-1)
    order: độ của filter
    """
    logger.info(f"Bắt đầu lưu dữ liệu filtered (cutoff={cutoff}, order={order})...")
    
    try:
        # Tạo folder processed
        base_dir = get_data_path.__globals__['BASE_DIR']
        processed_dir = base_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết kế lowpass filter
        b, a = butter(order, cutoff, btype='low')
        
        # Duyệt qua từng người
        for person in PEOPLE:
            # Tạo folder cho mỗi người
            person_dir = processed_dir / person
            person_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"\n📁 Xử lý người: {person}")
            
            # Duyệt qua từng file
            for file_num in DATA_FILES:
                path = get_data_path(person, file_num)
                df = pd.read_csv(path)
                
                # Lấy tất cả dữ liệu (tất cả samples)
                data = df.values.astype(float)
                
                # Áp dụng lowpass filter cho mỗi sample (mỗi hàng)
                filtered_data = np.zeros_like(data)
                for i in range(data.shape[0]):
                    filtered_data[i] = filtfilt(b, a, data[i])
                
                # Lưu vào CSV
                filtered_df = pd.DataFrame(filtered_data)
                output_path = person_dir / f"{file_num}.csv"
                filtered_df.to_csv(output_path, index=False)
                
                logger.info(f"  ✓ Lưu: {output_path} (shape: {filtered_data.shape})")
        
        logger.info(f"\n✓ Hoàn thành lưu dữ liệu filtered tại: {processed_dir}")
        print(f"\n{'='*70}")
        print(f"✅ Dữ liệu sau lowpass filter đã lưu tại: {processed_dir}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        logger.error(f"✗ Lỗi khi lưu dữ liệu: {e}")
        import traceback
        traceback.print_exc()

def ve_du_lieu_processed():
    """
    Vẽ dữ liệu đã filter từ folder 'processed'
    2 cột, 5 hàng (mỗi cột 5 người), mỗi file 1 màu (processed data)
    """
    logger.info("Bắt đầu vẽ dữ liệu từ folder processed...")
    
    try:
        base_dir = get_data_path.__globals__['BASE_DIR']
        processed_dir = base_dir / "processed"
        
        # Kiểm tra folder processed có tồn tại không
        if not processed_dir.exists():
            logger.error(f"✗ Folder 'processed' không tồn tại: {processed_dir}")
            return
        
        n_people = len(PEOPLE)
        fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
        
        # Màu sắc bình thường
        colors = plt.cm.tab10.colors
        
        # 🎨 Thiết lập style đẹp
        plt.style.use('seaborn-v0_8-darkgrid')
        fig.patch.set_facecolor('white')
        
        for i, person in enumerate(PEOPLE):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            ax.set_facecolor('#F8F9FA')
            
            for j, file_num in enumerate(DATA_FILES):
                path = processed_dir / person / f"{file_num}.csv"
                
                if not path.exists():
                    logger.warning(f"  ⚠️ Không tìm thấy: {path}")
                    continue
                
                df = pd.read_csv(path)
                
                # 🔥 lấy 1 sample (processed)
                sample = df.iloc[0].values.astype(float)
                
                ax.plot(sample, color=colors[j], 
                       label=f'Thickness {file_num}', linewidth=2.5, alpha=0.8)
            
            ax.set_title(f' {person.upper()}', fontsize=12, fontweight='bold', pad=10)
            ax.grid(alpha=0.25, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Cài đặt label cho trục
            ax.set_ylabel("Value", fontsize=10, fontweight='bold')
        
        # Cài đặt xlabel cho hàng cuối
        axes[4, 0].set_xlabel("Feature Index", fontsize=11, fontweight='bold')
        axes[4, 1].set_xlabel("Feature Index", fontsize=11, fontweight='bold')
        
        # Legend chung ở dưới cùng
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                  ncol=5, fontsize=10, framealpha=0.95, edgecolor='gray', fancybox=True)
        
        # Tiêu đề tổng quát
        plt.suptitle("Processed Data (Lowpass Filtered) - Sample 0 from Each File", 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        output_path = get_data_path.__globals__['PLOT_DIR'] / 'compare_processed_data.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        logger.info(f"✓ Đã lưu hình ảnh: {output_path}")
        
    except Exception as e:
        logger.error(f"Lỗi khi vẽ: {e}")
        import traceback
        traceback.print_exc()

def lay_du_lieu() -> dict:
    """
    Lấy dữ liệu từ tất cả file CSV
    Return: dict với key là tên người, value là list các DataFrame
    """
    logger.info("Bắt đầu lấy dữ liệu...")
    du_lieu = {}
    
    try:
        for person in PEOPLE:
            du_lieu[person] = []
            for file_num in DATA_FILES:
                path = get_data_path(person, file_num)
                df = pd.read_csv(path)
                du_lieu[person].append(df)
                logger.info(f"✓ Đã đọc: {path}")
        
        logger.info("✓ Hoàn thành lấy dữ liệu")
        return du_lieu
    
    except FileNotFoundError as e:
        logger.error(f"✗ Lỗi: Không tìm thấy file - {e}")
        return {}
    except Exception as e:
        logger.error(f"✗ Lỗi khi lấy dữ liệu: {e}")
        return {}


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

def tinh_mean_vector_va_cosine_similarity():
    """
    Tính mean vector cho mỗi người (sau normalize)
    Tính cosine similarity giữa các mean vectors
    Vẽ heatmap kết quả
    """
    logger.info("Bắt đầu tính mean vector và cosine similarity...")
    
    try:
        # Lấy dữ liệu
        du_lieu = lay_du_lieu()
        
        if not du_lieu:
            logger.error("✗ Lỗi: Không có dữ liệu")
            return
        
        # Tính mean vector cho mỗi người
        mean_vectors = {}
        
        for person in PEOPLE:
            if person not in du_lieu or not du_lieu[person]:
                continue
            
            # Ghép tất cả DataFrame của một người lại
            all_data = []
            for df in du_lieu[person]:
                data = df.values.astype(float)
                # Normalize mỗi sample
                normalized_data = normalize_data(data)
                all_data.append(normalized_data)
            
            # Ghép tất cả samples
            combined_data = np.vstack(all_data)
            
            # Tính mean vector
            mean_vector = np.mean(combined_data, axis=0)
            mean_vectors[person] = mean_vector
            
            logger.info(f"✓ Đã tính mean vector cho: {person} (shape: {mean_vector.shape})")
        
        # Tạo matrix từ mean vectors
        people_list = list(mean_vectors.keys())
        mean_matrix = np.array([mean_vectors[p] for p in people_list])
        
        # Tính cosine similarity
        cosine_sim = cosine_similarity(mean_matrix)
        
        # Vẽ heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cosine_sim, 
                   xticklabels=people_list,
                   yticklabels=people_list,
                   annot=True, 
                   fmt='.3f',
                   cmap='coolwarm',
                   center=0,
                   cbar_kws={'label': 'Cosine Similarity'},
                   linewidths=0.5,
                   square=True)
        
        plt.title('Cosine Similarity Heatmap - Mean Vectors of Each Person', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Lưu heatmap
        output_path = get_data_path.__globals__['PLOT_DIR'] / 'cosine_similarity_heatmap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        
        logger.info(f"✓ Đã lưu heatmap: {output_path}")
        
    except Exception as e:
        logger.error(f"✗ Lỗi khi tính toán: {e}")

def compare_people_per_label():
    """
    So sánh các NGƯỜI trong mỗi NHÃN (2, 4, 6, 8, 10)
    
    Với mỗi nhãn:
    - Lấy mean vector của từng người
    - Tính cosine similarity giữa các người
    - Vẽ heatmap
    
    Kết quả: 5 heatmap (một cho mỗi nhãn)
    """
    logger.info("Bắt đầu so sánh các người trong mỗi nhãn...")
    
    try:
        # Chuẩn bị thư mục lưu
        plot_dir = get_data_path.__globals__['PLOT_DIR'] / 'people_comparison'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Duyệt qua từng nhãn (file)
        for file_num in DATA_FILES:
            logger.info(f"\n📊 Phân tích nhãn {file_num}...")
            
            # Lấy mean vector cho mỗi người
            mean_vectors_people = {}
            
            for person in PEOPLE:
                path = get_data_path(person, file_num)
                df = pd.read_csv(path)
                
                # Normalize dữ liệu
                data = df.values.astype(float)
                normalized_data = normalize_data(data)
                
                # Tính mean vector
                mean_vector = np.mean(normalized_data, axis=0)
                mean_vectors_people[person] = mean_vector
                logger.info(f"  ✓ Lấy mean vector {person}: shape {mean_vector.shape}")
            
            # =============== Heatmap Cosine Similarity ===============
            people_sorted = sorted(mean_vectors_people.keys())
            mean_matrix = np.array([mean_vectors_people[p] for p in people_sorted])
            
            cosine_sim = cosine_similarity(mean_matrix)
            
            # Vẽ heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cosine_sim,
                       xticklabels=[p.upper() for p in people_sorted],
                       yticklabels=[p.upper() for p in people_sorted],
                       annot=True,
                       fmt='.3f',
                       cmap='RdYlGn',
                       vmin=0, vmax=1,
                       cbar_kws={'label': 'Cosine Similarity'},
                       linewidths=1,
                       square=True,
                       ax=ax)
            
            ax.set_title(f'Cosine Similarity Heatmap - Label {file_num}\n(So sánh các người)', 
                        fontsize=14, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            output_path = plot_dir / f'label_{file_num}_people_similarity.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"  ✓ Lưu: {output_path}")
            
            # =============== In kết quả ===============
            print(f"\n{'='*70}")
            print(f"NHÃN: {file_num}")
            print(f"{'='*70}")
            print("\n📈 Độ tương đồng (Cosine Similarity) giữa các người:")
            print("─" * 70)
            
            # Tạo bảng kết quả
            sim_df = pd.DataFrame(cosine_sim,
                                 index=[p.upper() for p in people_sorted],
                                 columns=[p.upper() for p in people_sorted])
            print(sim_df.to_string())
            
            # Tìm cặp người tương tự nhất
            # Lấy các giá trị similarity trừ đường chéo
            sim_copy = cosine_sim.copy()
            np.fill_diagonal(sim_copy, -1)  # Loại bỏ đường chéo
            
            max_idx = np.unravel_index(np.argmax(sim_copy), sim_copy.shape)
            min_idx = np.unravel_index(np.argmin(sim_copy), sim_copy.shape)
            
            print(f"\n✅ Cặp người tương tự NHẤT:      {people_sorted[max_idx[0]].upper()} <-> {people_sorted[max_idx[1]].upper()} ({cosine_sim[max_idx]:.3f})")
            print(f"❌ Cặp người khác nhau NHẤT:     {people_sorted[min_idx[0]].upper()} <-> {people_sorted[min_idx[1]].upper()} ({cosine_sim[min_idx]:.3f})")
            
            # Tính trung bình similarity
            mask = np.triu(np.ones_like(cosine_sim, dtype=bool), k=1)
            avg_sim = cosine_sim[mask].mean()
            print(f"📊 Trung bình similarity:        {avg_sim:.3f}")
        
        logger.info(f"\n✓ Hoàn thành phân tích! Tất cả heatmap lưu tại: {plot_dir}")
        print(f"\n{'='*70}")
        print("✅ Tất cả 5 heatmap đã được tạo thành công!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        logger.error(f"✗ Lỗi khi so sánh: {e}")
        import traceback
        traceback.print_exc()