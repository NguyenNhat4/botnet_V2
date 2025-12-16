# Báo cáo Phân tích Mã nguồn Dự án Botnet Detection

Tài liệu này được biên soạn nhằm cung cấp cái nhìn sâu sắc về kiến trúc, luồng dữ liệu và logic huấn luyện của hệ thống phát hiện Botnet. Tài liệu tập trung vào các chi tiết kỹ thuật cần thiết để Cố vấn Kỹ thuật đánh giá và đưa ra chiến lược huấn luyện hiệu quả.

## 1. Tổng quan Kiến trúc (System Overview)

Dự án xây dựng một hệ thống phân loại luồng mạng (Network Flow Classification) sử dụng kiến trúc **Deep Learning (1D-CNN)**.

*   **Dữ liệu đầu vào:** File `.binetflow` từ tập dữ liệu CTU-13, được chuyển đổi sang `.csv`.
*   **Mục tiêu:** Phân loại luồng thành 3 lớp: `Botnet`, `C&C`, `Normal`.
*   **Thách thức chính được xử lý:**
    *   Dữ liệu mất cân bằng (Imbalanced Data).
    *   Đặc trưng có không gian mẫu lớn (High cardinality) như IP, Port.
    *   Sự chênh lệch về quy mô giá trị (Feature scaling).

---

## 2. Phân hệ Xử lý Dữ liệu (Data Pipeline)

Đây là phần quan trọng nhất ảnh hưởng đến chất lượng mô hình. Logic xử lý nằm chủ yếu trong `preprocessing_utils.py` và `data_loader.py`.

### 2.1. Chiến lược "Global Statistics" (`train.py`, `preprocessing_utils.py`)
Trước khi training, hệ thống thực hiện một bước **Pre-scan** toàn bộ dữ liệu huấn luyện để xây dựng các thống kê toàn cục. Điều này đảm bảo tính nhất quán giữa các batch và giữa tập Train/Test.

*   **Tần suất IP/Port (`calculate_global_frequencies`):**
    *   Thay vì dùng One-Hot Encoding cho IP (gây bùng nổ chiều dữ liệu) hay Hash, hệ thống đếm tần suất xuất hiện của `SrcAddr`, `DstAddr`, `Sport`, `Dport` trên toàn bộ tập Train.
    *   Giá trị các cột này được thay thế bằng tần suất của chúng (`Src_freq`, `Dst_freq`, ...).
*   **Top Trạng thái kết nối (`top_states`):**
    *   Cột `State` có rất nhiều giá trị lạ. Hệ thống chỉ giữ lại **Top N** (mặc định 15 - cấu hình trong `config.py`) trạng thái phổ biến nhất. Các trạng thái còn lại gộp thành `Other`.
*   **Đồng bộ cột (`expected_columns`):**
    *   Xác định danh sách cột cố định sau khi One-Hot Encoding để đảm bảo mọi batch dữ liệu đều có shape giống nhau (thêm cột thiếu bằng 0).

### 2.2. Tiền xử lý chi tiết (`preprocessing_utils.process_batch_fast_v2`)
Quy trình xử lý một chunk dữ liệu thô:

1.  **Gán nhãn (`quick_classify`):**
    *   Chuỗi chứa "botnet" $\rightarrow$ `Botnet`
    *   Chuỗi chứa "c&c" hoặc "cc" $\rightarrow$ `C&C`
    *   Khác $\rightarrow$ `Normal`
2.  **Làm sạch:** Loại bỏ cột `StartTime`. Chuyển đổi `Dur`, `TotPkts`, `TotBytes`, `SrcBytes`, `Sport`, `Dport` sang dạng số.
3.  **Feature Engineering:**
    *   Thay thế IP/Port bằng tần suất (Frequency Encoding).
    *   Giữ lại `Sport`, `Dport` gốc dưới dạng số (Model học cả giá trị Port và độ phổ biến của Port).
    *   **One-Hot Encoding:** Áp dụng cho `Proto` (Giao thức), `Dir` (Hướng), và `State` (Top N).
4.  **Xử lý ngoại lệ:** Điền `NaN` bằng 0.

### 2.3. Chuẩn hóa (`data_loader.py`)
Hệ thống áp dụng phương pháp chuẩn hóa mạnh mẽ để xử lý các phân phối lệch (skewed data) thường thấy trong mạng máy tính (ví dụ: số byte cực lớn):

1.  **Log Transformation:** Áp dụng `np.log1p(x)` (logarit tự nhiên của x + 1) cho **toàn bộ đặc trưng**. Điều này giúp thu gọn khoảng giá trị của các đặc trưng có đuôi dài (long-tail) như `TotBytes`.
2.  **Robust Scaler:** Sử dụng `sklearn.preprocessing.RobustScaler` (trừ trung vị và chia cho khoảng tứ phân vị IQR). Cách này giúp mô hình ít bị ảnh hưởng bởi outliers hơn so với StandardScaler. Scaler được `fit` trên tập Train và `transform` trên tập Test.

---

## 3. Mô hình & Huấn luyện (Model & Training)

### 3.1. Kiến trúc Mô hình (`model.py`)
Sử dụng mạng Convolutional 1 chiều (**1D-CNN**), phù hợp với dữ liệu dạng bảng nhưng muốn khai thác mối quan hệ tương quan cục bộ giữa các đặc trưng (sau khi đã sắp xếp cột cố định).

*   **Input:** Vector 1 chiều (Batch Size, Channels=1, Features).
*   **Layer 1:** Conv1d (64 filters, k=5) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ Dropout(0.3). **Lưu ý:** Không có Pooling ở đây để giữ thông tin chi tiết ban đầu.
*   **Layer 2:** Conv1d (128 filters, k=3) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ Dropout(0.3) $\rightarrow$ **MaxPool1d(2)**.
*   **Layer 3:** Conv1d (256 filters, k=3) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ Dropout(0.3).
*   **Output Block:** Global Average Pooling $\rightarrow$ Flatten $\rightarrow$ Dense(128) $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(3 classes).

### 3.2. Chiến lược Loss & Imbalance (`loss.py`, `train.py`)
Dữ liệu Botnet thường rất mất cân bằng (Normal chiếm đa số). Hệ thống giải quyết bằng 2 cách kết hợp:

1.  **Class Weights:** Tính toán trọng số lớp (`sklearn.utils.class_weight.compute_class_weight`) dựa trên dữ liệu Train ("balanced").
2.  **Focal Loss:** Sử dụng hàm loss tùy chỉnh `FocalLoss` thay vì CrossEntropy thuần túy.
    *   **Cơ chế:** Giảm trọng số của các mẫu dễ phân loại (xác suất dự đoán cao), tập trung vào các mẫu khó (hard examples).
    *   **Tham số:** `gamma=4.0` (tập trung rất mạnh vào mẫu khó), kết hợp với `weight` từ bước trên.

### 3.3. Quy trình Huấn luyện (`train.py`)
*   **Optimizer:** Adam (`lr=0.0001`).
*   **Batch Size:** 256.
*   **Epochs:** 6.
*   **Validation:** Tách 20% từ tập Train (stratified split) để làm Validation set.
*   **Checkpoint:** Lưu model (`best_model.pth`) mỗi khi Validation Loss giảm.
*   **Không sử dụng Sampler:** Code có import `WeightedRandomSampler` nhưng hiện tại **không sử dụng** trong `DataLoader` (shuffle=True). Việc cân bằng dựa hoàn toàn vào Loss function.

---

## 4. Cấu hình (`config.py`)
Nơi tập trung các tham số có thể tinh chỉnh để tối ưu hóa:

*   `TRAIN_SCENARIOS`: Danh sách ID các kịch bản dùng để train (ví dụ: `['3', '4', '11']`).
*   `TEST_SCENARIOS`: Danh sách ID kịch bản test (ví dụ: `['10']`).
*   `STATE_TOP_N`: Số lượng trạng thái 'State' giữ lại (mặc định 15).
*   `BATCH_SIZE`, `LEARNING_RATE`.

---

## 5. Công cụ Đánh giá (`evaluate.py` & `analyze_csv.py`)

### 5.1. `evaluate.py`
Dùng để kiểm thử model đã train trên tập dữ liệu mới.
*   Tự động tải `global_stats.pkl` và `scaler.pkl` cũ để đảm bảo xử lý dữ liệu Test y hệt Train.
*   Báo cáo: Accuracy, Precision, Recall, F1 (Weighted) và Confusion Matrix.
*   Lưu kết quả chi tiết vào thư mục `metrics/`.

### 5.2. `analyze_csv.py`
Công cụ độc lập giúp Developer/Data Scientist hiểu dữ liệu trước khi train.
*   Đầu vào: File CSV bất kỳ.
*   Đầu ra:
    *   Phân bố nhãn (Mapped & Raw).
    *   Thống kê mô tả (Min, Max, Mean, Std) của các cột số.
    *   Biểu đồ phân bố (Boxplot Log-scale, Bar chart).
*   **Lời khuyên:** Dùng tool này để kiểm tra xem một scenario mới có phân bố quá khác biệt so với tập train hay không (Data Drift).

---

## 6. Lời khuyên cho Cố vấn (Advisor Notes)

Dựa trên code hiện tại, một số điểm cần lưu ý khi đưa ra chiến lược training:

1.  **Dữ liệu:** Hệ thống phụ thuộc lớn vào **tần suất IP**. Nếu mạng Botnet thay đổi IP liên tục hoặc tấn công từ dải IP hoàn toàn mới chưa từng xuất hiện trong tập Train, hiệu năng có thể giảm (IP mới sẽ có tần suất = 0 hoặc thấp). $\rightarrow$ Cần đánh giá kỹ khả năng Generalization trên các Scenario có IP lạ.
2.  **Chuẩn hóa:** Việc dùng `np.log1p` trước `RobustScaler` là một chiến lược tốt cho dữ liệu mạng. Đừng bỏ bước này.
3.  **Mô hình:** `Dropout` khá dày (0.3 - 0.5) giúp tránh Overfitting, nhưng với 1D-CNN, vị trí cột (thứ tự đặc trưng) là cố định.
4.  **Cân bằng lớp:** Hiện tại chỉ dựa vào `FocalLoss`. Nếu model vẫn thiên vị lớp Normal, cân nhắc kích hoạt `WeightedRandomSampler` trong `train.py` (đã import nhưng chưa dùng).
