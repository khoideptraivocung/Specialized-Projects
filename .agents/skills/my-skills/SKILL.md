---
name: auto-ml-thickness-pipeline
description: Tự động train, evaluate và tối ưu model dự đoán độ dày từ dữ liệu time-series. Dùng khi cần build pipeline ML hoàn chỉnh từ raw data đến model tốt nhất.
---

# 🤖 Auto ML Thickness Pipeline Skill

Skill này giúp agent tự động:
- Load dữ liệu từ folder
- Split theo người (8 train, 1 val, 1 test)
- Preprocess (filter + outlier detection + normalize)
- Train model CNN
- Evaluate + visualize
- Lưu kết quả + config + model
- Tự cải thiện qua nhiều lần train

⚠️ Tất cả output phải bằng **TIẾNG VIỆT**

---
## 🔧 Execution

Agent phải:

1. Tìm file:

scripts/run_pipeline.py

2. Nếu tồn tại:
- Import và chạy:
```python
from scripts.run_pipeline import run
run()
```


# 📌 Khi nào dùng skill này

- Khi cần train model regression từ dữ liệu sensor/time-series
- Khi muốn pipeline tự động end-to-end
- Khi cần so sánh nhiều lần train (experiment tracking)
- Khi dữ liệu có nhiều người (multi-subject)

---
## Nếu không tồn tại:
- Tự tạo file theo pipeline đã mô tả
- Sau đó chạy lại

# ⚙️ Cách sử dụng

## 1. Đọc cấu trúc dữ liệu

Agent phải:
- Tự detect folder data
- Mỗi người = 1 folder hoặc 1 nhóm file
- Label = độ dày (2,4,6,8,10)

Nếu chưa có cấu trúc:

data/
raw/
processed/


---

## 2. Split dữ liệu (QUAN TRỌNG)

- Train: 8 người
- Validation: 1 người
- Test: 1 người

❗ Không được split theo sample

---

## 3. Preprocessing pipeline

### 3.1 Low-pass filter
- Áp dụng cho toàn bộ tín hiệu

### 3.2 Isolation Forest
- contamination = 0.05
- Fit trên TRAIN
- Apply cho VAL + TEST

### 3.3 Normalize (Z-score)
- Fit trên TRAIN
- Transform VAL + TEST

---

## 4. Chuẩn bị dữ liệu

- X shape: (N, 333)
- CNN input: reshape → (N, 333, 1)
- y: float (độ dày)

---

## 5. Model

### Loss (bắt buộc)

MSE


### Metric (tùy chọn)
- MAE (an toàn)
- MAPE (chỉ dùng khi y không gần 0)

---

## 6. Training

- Optimizer: Adam
- Epoch: 50–200
- Batch size: 16–64

### Callback
- EarlyStopping
- ReduceLROnPlateau

---

## 7. Evaluation (BẮT BUỘC)

### 7.1 Metrics
- MSE
- MAE hoặc MAPE

---

### 7.2 Visualization

#### 1. Scatter plot
- y_true vs y_pred

#### 2. Residual plot

residual = y_true - y_pred

- Plot theo y_pred

👉 Detect:
- Bias
- Pattern lỗi

#### 3. Error bars
- Group theo label (2,4,6,8,10)
- Tính:
  - mean prediction
  - std

Plot:
- x = label
- y = mean
- error = std

👉 Detect:
- Độ ổn định model

---

## 8. Nếu scale y

Phải:

inverse_transform trước khi evaluate


---

## 9. Lưu kết quả (BẮT BUỘC)

Mỗi lần train:

### 9.1 Tạo experiment_id

exp_001
exp_002


### 9.2 Lưu vào:

experiments/exp_xxx/


### 9.3 Bao gồm:
- model.keras
- scaler_X.pkl
- scaler_y.pkl (nếu có)
- history.json
- metrics.json
- scatter.png
- residual.png
- error_bar.png

---

## 10. Tự tạo config.yaml

Ví dụ:

```yaml
data:
  split: person
  train: 8
  val: 1
  test: 1

preprocess:
  lowpass: true
  isolation_forest:
    contamination: 0.05
  normalize: zscore

model:
  type: cnn
  input_shape: [333, 1]

training:
  loss: mse
  metric: mae
  epochs: 100
  batch_size: 32
```


---

## 11. Tự tạo README.md

### Bao gồm:
- Mô tả bài toán  
- Pipeline  
- Cách chạy  
- Kết quả tốt nhất  

---

## 12. Tự tạo requirements.txt
- numpy
- pandas
- scikit-learn
- tensorflow
- matplotlib


---

## 13. Auto Improvement Loop

Lặp:

1. Train  
2. Evaluate  
3. So sánh với best  
4. Nếu tốt hơn → lưu best  
5. Nếu không → thử:
   - giảm learning rate  
   - đổi kiến trúc  
   - thêm dropout  

---

## 14. Tiêu chí thành công

- MAE hoặc MAPE thấp (tùy theo dữ liệu cụ thể, ví dụ ~ < 1)
- Train ≈ Validation  
- Scatter gần đường y = x  
- Residual không có pattern  

---
## 15. Quy tắc ra quyết định (Decision Rules)

- Nếu overfitting → tăng dropout hoặc giảm model size  
- Nếu underfitting → tăng số filter / layer  
- Nếu loss không giảm → giảm learning rate  
- Nếu MAPE cao bất thường → kiểm tra scale y  
---
# 🚀 Tóm tắt pipeline


Load → Split(person) → Filter → Outlier → Normalize
→ Train → Evaluate → Save → Improve


---

# ⚠️ Nguyên tắc vàng

- Data quan trọng hơn model  
- Không được leakage  
- Luôn fit trên train  
- Mọi output = tiếng Việt
- Sử dụng ít các emoji không cần thiết  

---
## 🔁 Auto Execution Loop

- Sau khi chạy xong:
  - So sánh với best model
  - Nếu chưa tốt:
    - sửa config.yaml
    - chạy lại run_pipeline.py