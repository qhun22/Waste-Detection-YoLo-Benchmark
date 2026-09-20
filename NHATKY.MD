# NHẬT KÝ PHÁT TRIỂN DỰ ÁN (PROJECT DEV LOG)

## Đề tài

**Phân loại rác thải tự động hỗ trợ phân loại nguồn bằng Deep Learning**

## Mô hình nghiên cứu

-   Ultralytics YOLO
-   YOLOv5s
-   YOLOv8n
-   YOLOv8s
-   YOLO11n

## Bộ dữ liệu

**GARBAGE-CLASSIFICATION-3 (v2 GC1)**

-   Tổng số ảnh: **10.464 ảnh**
-   Định dạng dữ liệu: YOLO
-   Số lớp: **6 lớp**

## Môi trường thực nghiệm

### Môi trường Local

-   Thiết bị: Laptop ASUS Vivobook 15 (X1505VA)
-   Hệ điều hành: Windows 11 64-bit
-   Trình soạn thảo: VS Code
-   Python: 3.14
-   Môi trường ảo: `.venv`

### Môi trường Cloud Computing

-   Nền tảng: Google Colab
-   GPU: NVIDIA Tesla T4
-   VRAM: 16GB
-   CUDA: 13.0
-   PyTorch: 2.11.0
-   Ultralytics: 8.4.155

------------------------------------------------------------------------

# GIAI ĐOẠN 1: THIẾT LẬP MÔI TRƯỜNG VÀ CẤU TRÚC DỰ ÁN LOCAL

## 1.1. Khởi tạo cấu trúc thư mục

Dự án được tổ chức theo cấu trúc tương đối rõ ràng, phù hợp với quy
trình phát triển và thực nghiệm Machine Learning/Deep Learning.

``` text
project-root/
│
├── configs/
│   └── Các file cấu hình tham số, hyperparameter,
│       ánh xạ nhãn nghiệp vụ
│
├── data/
│   ├── raw/
│   │   └── Dữ liệu thô ban đầu
│   │
│   └── dataset/
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── data.yaml
│
├── notebooks/
│   └── Các notebook EDA và kiểm thử nhanh
│
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils/
│       ├── download_roboflow.py
│       └── check_labels.py
│
├── results/
│   ├── weights/
│   ├── figures/
│   └── predictions/
│
├── requirements.txt
└── .gitignore
```

### Vai trò của các thư mục

  -----------------------------------------------------------------------
  Thư mục/File                        Chức năng
  ----------------------------------- -----------------------------------
  `configs/`                          Lưu cấu hình tham số,
                                      hyperparameter và ánh xạ nhãn
                                      nghiệp vụ

  `data/raw/`                         Lưu dữ liệu thô ban đầu

  `data/dataset/`                     Lưu bộ dữ liệu đã được chuẩn hóa để
                                      huấn luyện

  `notebooks/`                        Thực hiện EDA, trực quan hóa và
                                      kiểm thử nhanh

  `src/`                              Chứa mã nguồn chính của dự án

  `src/utils/`                        Chứa các tiện ích hỗ trợ tải và
                                      kiểm tra dữ liệu

  `results/weights/`                  Lưu các file trọng số mô hình

  `results/figures/`                  Lưu biểu đồ, ma trận nhầm lẫn và
                                      kết quả đánh giá

  `results/predictions/`              Lưu ảnh dự đoán sau suy luận

  `.gitignore`                        Loại trừ môi trường ảo, dữ liệu và
                                      các file dung lượng lớn
  -----------------------------------------------------------------------

## 1.2. Thiết lập môi trường ảo

Môi trường ảo được tạo bằng lệnh:

``` powershell
python -m venv .venv
```

Việc sử dụng môi trường ảo giúp tách biệt các thư viện của dự án khỏi
môi trường Python chung trên máy tính, hạn chế xung đột phiên bản và
thuận tiện khi tái lập môi trường.

## 1.3. Xử lý lỗi quyền thực thi PowerShell

Trong quá trình kích hoạt môi trường ảo, PowerShell phát sinh lỗi do
chính sách bảo mật ngăn không cho chạy file script `Activate.ps1`.

Lệnh được sử dụng để mở quyền thực thi trong phạm vi phiên làm việc hiện
tại:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Sau đó, môi trường ảo được kích hoạt thành công:

``` powershell
.venv\Scripts\Activate.ps1
```

Khi kích hoạt thành công, terminal hiển thị tiền tố:

``` text
(.venv)
```

## 1.4. Cài đặt thư viện phụ thuộc

Các thư viện được khai báo trong file `requirements.txt`.

### Nhóm mô hình và tính toán

-   `ultralytics`
-   `torch`
-   `torchvision`
-   `roboflow`

### Nhóm xử lý ảnh và dữ liệu

-   `opencv-python`
-   `pillow`
-   `numpy`
-   `pandas`

### Nhóm trực quan hóa và tiện ích

-   `matplotlib`
-   `seaborn`
-   `pyyaml`
-   `tqdm`

Mục đích của các thư viện là hỗ trợ toàn bộ quy trình từ tải dữ liệu,
tiền xử lý, huấn luyện, đánh giá đến suy luận mô hình.

------------------------------------------------------------------------

# GIAI ĐOẠN 2: THU THẬP, TIỀN XỬ LÝ VÀ PHÂN TÍCH DỮ LIỆU (EDA)

## 2.1. Thu thập dữ liệu

Bộ dữ liệu được thu thập thông qua nền tảng Roboflow bằng API.

Script thực hiện tải dữ liệu:

``` text
src/utils/download_roboflow.py
```

Dataset được tải thành công với tên:

``` text
GARBAGE CLASSIFICATION 3 (v2 GC1)
```

Dữ liệu được xuất theo định dạng YOLOv8. Định dạng này có thể được sử
dụng chung cho các mô hình YOLOv5, YOLOv8 và YOLO11 trong phạm vi thực
nghiệm của dự án.

## 2.2. Chuẩn hóa cấu hình dữ liệu

File cấu hình dữ liệu:

``` text
data/dataset/data.yaml
```

Nội dung cấu hình chính:

``` yaml
path: data/dataset
train: train/images
val: valid/images
test: test/images

nc: 6

names:
  - BIODEGRADABLE
  - CARDBOARD
  - GLASS
  - METAL
  - PAPER
  - PLASTIC
```

### Ý nghĩa cấu hình

  Thuộc tính   Ý nghĩa
  ------------ ------------------------------
  `path`       Đường dẫn gốc của bộ dữ liệu
  `train`      Đường dẫn đến ảnh huấn luyện
  `val`        Đường dẫn đến ảnh validation
  `test`       Đường dẫn đến ảnh kiểm thử
  `nc`         Số lượng lớp đối tượng
  `names`      Danh sách tên các lớp

## 2.3. Kiểm định tính toàn vẹn dữ liệu

Script kiểm tra dữ liệu:

``` text
src/utils/check_labels.py
```

Script được sử dụng để kiểm tra:

-   Số lượng ảnh trong từng tập dữ liệu.
-   Sự tồn tại của file nhãn.
-   Tính hợp lệ của tọa độ bounding box.
-   Phân phối dữ liệu giữa các tập.
-   Khả năng đọc ảnh và nhãn.

### Phân chia dữ liệu

  Tập dữ liệu       Số lượng ảnh        Tỷ lệ
  --------------- -------------- ------------
  Train                    7.324   Khoảng 70%
  Validation               2.098   Khoảng 20%
  Test                     1.042   Khoảng 10%
  **Tổng cộng**       **10.464**     **100%**

## 2.4. Trực quan hóa bounding box

Một số ảnh mẫu được trực quan hóa bounding box để kiểm tra sự khớp nhau
giữa:

-   Ảnh đầu vào.
-   File nhãn YOLO.
-   Tọa độ tâm bounding box.
-   Chiều rộng và chiều cao bounding box.

Ảnh trực quan hóa được lưu tại:

``` text
results/figures/dataset_sample.png
```

Việc kiểm tra này giúp phát hiện sớm các lỗi như:

-   Bounding box lệch khỏi đối tượng.
-   Tọa độ không nằm trong khoảng hợp lệ.
-   Nhãn không tương ứng với ảnh.
-   File ảnh hoặc file nhãn bị thiếu.

## 2.5. Thiết kế phân lớp nghiệp vụ

Trong quá trình huấn luyện, dự án giữ nguyên 6 lớp gốc để mô hình có thể
học được nhiều đặc trưng chi tiết hơn.

### Sáu lớp huấn luyện

1.  `BIODEGRADABLE`
2.  `CARDBOARD`
3.  `GLASS`
4.  `METAL`
5.  `PAPER`
6.  `PLASTIC`

### Ánh xạ sang nhóm quản lý

Ở tầng suy luận, kết quả được gom về các nhóm phục vụ mục đích phân loại
rác.

  -----------------------------------------------------------------------
  Nhóm quản lý                        Các lớp thành phần
  ----------------------------------- -----------------------------------
  Rác hữu cơ                          `BIODEGRADABLE`

  Rác tái chế / vô cơ                 `CARDBOARD`, `GLASS`, `METAL`,
                                      `PAPER`, `PLASTIC`

  Rác nguy hại / đặc thù              Quy hoạch mở rộng cho thiết bị điện
                                      tử, pin
  -----------------------------------------------------------------------

Lưu ý: nhóm rác nguy hại / đặc thù hiện là hướng mở rộng trong tương
lai, chưa nằm trong 6 lớp dữ liệu được huấn luyện ở thực nghiệm này.

------------------------------------------------------------------------

# GIAI ĐOẠN 3: HUẤN LUYỆN VÀ ĐỐI SÁNH THỰC NGHIỆM TRÊN CLOUD GPU

## 3.1. Thiết lập thực nghiệm chung

Để việc so sánh giữa các mô hình có tính nhất quán, các thử nghiệm được
cố định chung bộ siêu tham số:

  Tham số      Giá trị
  ------------ -----------------------
  Epochs       30
  Image size   640
  Batch size   32
  Optimizer    AdamW
  Device       `0` - NVIDIA Tesla T4

Các mô hình được huấn luyện và đánh giá trên cùng bộ dữ liệu với cùng
cấu hình thực nghiệm.

------------------------------------------------------------------------

## 3.2. Thực nghiệm với YOLOv5s

### Thông tin mô hình

-   Mô hình: `yolov5su.pt`
-   Vai trò: Baseline
-   Kiến trúc: Anchor-based
-   Số layers: 153
-   Số tham số: 9.113.858
-   Độ phức tạp: 23.8 GFLOPs

### Tài nguyên và thời gian

-   Thời gian huấn luyện: 1.091 giờ, tương đương khoảng 65 phút.
-   VRAM tiêu hao: khoảng 8.0 GB.
-   Dung lượng file trọng số: 18.5 MB.

### Kết quả validation tại Epoch 30

  Chỉ số        Kết quả
  ----------- ---------
  Precision       57.9%
  Recall          45.9%
  mAP@50          52.3%
  mAP@50-95       35.6%

### Tốc độ suy luận trên Tesla T4

  Giai đoạn       Thời gian
  ------------- -----------
  Preprocess         0.5 ms
  Inference          5.5 ms
  Postprocess        1.7 ms

### Hiệu năng theo lớp - mAP@50

  Lớp               mAP@50
  --------------- --------
  GLASS              80.2%
  METAL              68.0%
  BIODEGRADABLE      61.9%
  CARDBOARD          59.3%
  PLASTIC            37.3%
  PAPER               6.8%

------------------------------------------------------------------------

## 3.3. Thực nghiệm với YOLOv8n

### Thông tin mô hình

-   Mô hình: `yolov8n.pt`
-   Vai trò: Lightweight
-   Kiến trúc: Anchor-free
-   Số layers: 129, fused 72
-   Số tham số: 3.006.818
-   Độ phức tạp: 8.1 GFLOPs

### Tài nguyên và thời gian

-   Thời gian huấn luyện: 0.968 giờ, tương đương khoảng 58 phút.
-   VRAM tiêu hao: 6.25 GB.
-   Dung lượng file trọng số: 6.3 MB.

### Kết quả validation tại Epoch 30

  Chỉ số        Kết quả
  ----------- ---------
  Precision       56.7%
  Recall          44.9%
  mAP@50          50.9%
  mAP@50-95       34.5%

### Tốc độ suy luận trên Tesla T4

  Giai đoạn       Thời gian
  ------------- -----------
  Preprocess         0.7 ms
  Inference          2.6 ms
  Postprocess        1.8 ms

YOLOv8n có thời gian inference thấp nhất trong bốn mô hình được thử
nghiệm.

### Hiệu năng theo lớp - mAP@50

  Lớp               mAP@50
  --------------- --------
  GLASS              78.4%
  METAL              64.3%
  BIODEGRADABLE      60.7%
  CARDBOARD          59.3%
  PLASTIC            36.1%
  PAPER              6.98%

------------------------------------------------------------------------

## 3.4. Thực nghiệm với YOLOv8s

### Thông tin mô hình

-   Mô hình: `yolov8s.pt`
-   Vai trò: Accuracy-focused
-   Kiến trúc: Anchor-free
-   Số layers: 129, fused 72
-   Số tham số: 11.127.906
-   Độ phức tạp: 28.4 GFLOPs

### Tài nguyên và thời gian

-   Thời gian huấn luyện: 1.122 giờ, tương đương khoảng 67 phút.
-   VRAM tiêu hao: khoảng 8.5--9.7 GB.
-   Dung lượng file trọng số: 22.5 MB.

### Kết quả validation tại Epoch 30

  Chỉ số        Kết quả
  ----------- ---------
  Precision       57.9%
  Recall          47.8%
  mAP@50          54.1%
  mAP@50-95       37.1%

### Tốc độ suy luận trên Tesla T4

  Giai đoạn       Thời gian
  ------------- -----------
  Preprocess         0.6 ms
  Inference          6.5 ms
  Postprocess        1.9 ms

### Hiệu năng theo lớp - mAP@50

  Lớp               mAP@50
  --------------- --------
  GLASS              81.0%
  METAL              68.7%
  BIODEGRADABLE      62.5%
  CARDBOARD          60.6%
  PLASTIC            44.3%
  PAPER              7.19%

------------------------------------------------------------------------

## 3.5. Thực nghiệm với YOLO11n

### Thông tin mô hình

-   Mô hình: `yolo11n.pt`
-   Vai trò: Mô hình thuộc thế hệ YOLO mới
-   Kiến trúc: C3k2 và C2PSA
-   Số layers: 181, fused 100
-   Số tham số: 2.583.322
-   Độ phức tạp: 6.4 GFLOPs

### Tài nguyên và thời gian

-   Thời gian huấn luyện: 1.014 giờ, tương đương khoảng 61 phút.
-   VRAM tiêu hao: 6.87 GB.
-   Dung lượng file trọng số: 5.5 MB.

### Kết quả validation tại Epoch 30

  Chỉ số        Kết quả
  ----------- ---------
  Precision       58.7%
  Recall          46.1%
  mAP@50          51.4%
  mAP@50-95       34.9%

### Tốc độ suy luận trên Tesla T4

  Giai đoạn       Thời gian
  ------------- -----------
  Preprocess         0.7 ms
  Inference          6.2 ms
  Postprocess        2.5 ms

### Hiệu năng theo lớp - mAP@50

  Lớp               mAP@50
  --------------- --------
  GLASS              80.9%
  METAL              66.5%
  BIODEGRADABLE      61.0%
  CARDBOARD          58.2%
  PLASTIC            34.3%
  PAPER              7.62%

------------------------------------------------------------------------

# BẢNG ĐỐI SÁNH TỔNG THỂ (MASTER BENCHMARK)

  ------------------------------------------------------------------------
  Tiêu chí           YOLOv5s         YOLOv8n        YOLOv8s        YOLO11n
  đối sánh        (Baseline)   (Lightweight)     (High-Acc) 
  ----------- -------------- --------------- -------------- --------------
  Kiến trúc               C3             C2f            C2f   C3k2 + C2PSA
  Backbone                                                  

  Kiểu gán      Anchor-based     Anchor-free    Anchor-free    Anchor-free
  nhãn                                                      

  Số tham số          9.11 M          3.01 M        11.13 M     **2.58 M**

  Độ phức tạp         23.8 G           8.1 G         28.4 G      **6.4 G**

  Dung lượng         18.5 MB          6.3 MB        22.5 MB     **5.5 MB**
  Weight                                                    

  VRAM tiêu         \~8.0 GB     **6.25 GB**       \~9.2 GB        6.87 GB
  hao                                                       

  Thời gian          1.091 h     **0.968 h**        1.122 h        1.014 h
  train - 30                                                
  epoch                                                     

  Precision            57.9%           56.7%          57.9%      **58.7%**

  Recall               45.9%           44.9%      **47.8%**          46.1%

  mAP@50               52.3%           50.9%      **54.1%**          51.4%

  mAP@50-95            35.6%           34.5%      **37.1%**          34.9%

  Inference           5.5 ms      **2.6 ms**         6.5 ms         6.2 ms
  Time                                                      
  ------------------------------------------------------------------------

## Nhận xét tổng quan từ bảng benchmark

-   **YOLOv8s** đạt mAP@50 và mAP@50-95 cao nhất trong bốn mô hình.
-   **YOLO11n** có số tham số, FLOPs và dung lượng trọng số thấp nhất.
-   **YOLOv8n** có thời gian inference thấp nhất trên Tesla T4.
-   **YOLO11n** đạt Precision cao nhất.
-   **YOLOv8s** đạt Recall cao nhất.
-   Lớp `GLASS` có kết quả tốt nhất ở cả bốn mô hình.
-   Lớp `PAPER` có kết quả thấp nhất, cho thấy đây là lớp cần được xem
    xét để cải thiện dữ liệu hoặc phương pháp huấn luyện.

Các kết quả trên phản ánh hiệu năng của các mô hình trong điều kiện thực
nghiệm cụ thể: 30 epoch, kích thước ảnh 640, batch size 32 và GPU NVIDIA
Tesla T4. Do đó, kết quả có thể thay đổi nếu sử dụng số epoch, dữ liệu
hoặc cấu hình khác.

------------------------------------------------------------------------

# GIAI ĐOẠN 4: HẬU HUẤN LUYỆN, ĐỒNG BỘ ARTIFACTS VÀ INFERENCE

## 4.1. Đồng bộ artifacts từ Google Colab

Sau khi hoàn thành quá trình huấn luyện, toàn bộ kết quả thực nghiệm
được đóng gói thành file:

``` text
all_waste_benchmarks.zip
```

-   Dung lượng gói nén: khoảng 128 MB.
-   Nguồn tải: Google Colab.
-   Mục đích: Đồng bộ trọng số, biểu đồ đánh giá và kết quả thực nghiệm
    về máy local.

## 4.2. Phân bổ file trọng số

Các file trọng số tối ưu được phân bổ vào thư mục:

``` text
results/weights/
```

Danh sách file:

``` text
results/
└── weights/
    ├── yolov5s_best.pt
    ├── yolov8n_best.pt
    ├── yolov8s_best.pt
    └── yolo11n_best.pt
```

## 4.3. Đồng bộ biểu đồ và kết quả đánh giá

Các artifact trực quan được đồng bộ vào:

``` text
results/figures/
```

Các loại file chính bao gồm:

-   `confusion_matrix.png`
-   `BoxPR_curve.png`
-   `results.png`

Những file này được lưu tương ứng cho cả bốn mô hình, phục vụ việc phân
tích:

-   Ma trận nhầm lẫn.
-   Precision-Recall.
-   Diễn biến loss qua các epoch.
-   Diễn biến các chỉ số đánh giá.
-   So sánh hiệu quả huấn luyện giữa các mô hình.

## 4.4. Xây dựng module suy luận

Module suy luận chính:

``` text
src/inference.py
```

Chức năng của module:

1.  Tải file trọng số tối ưu.
2.  Đọc ảnh từ tập kiểm thử.
3.  Thực hiện nhận diện đối tượng.
4.  Lấy nhãn, độ tin cậy và tọa độ bounding box.
5.  Ánh xạ 6 lớp gốc sang nhóm quản lý.
6.  Vẽ bounding box lên ảnh.
7.  Lưu ảnh kết quả vào thư mục dự đoán.

Trong phiên bản suy luận hiện tại, mô hình được lựa chọn để thực hiện
nhận diện là:

``` text
yolov8s_best.pt
```

Dữ liệu đầu vào:

``` text
data/dataset/test/images
```

Thư mục lưu kết quả:

``` text
results/predictions/
```

## 4.5. Ánh xạ kết quả suy luận

Kết quả từ mô hình gồm sáu lớp:

``` text
BIODEGRADABLE
CARDBOARD
GLASS
METAL
PAPER
PLASTIC
```

Sau đó được ánh xạ thành các nhóm nghiệp vụ:

``` text
BIODEGRADABLE
    └── Rác Hữu Cơ

CARDBOARD
GLASS
METAL
PAPER
PLASTIC
    └── Rác Tái Chế / Vô Cơ
```

Nhóm rác nguy hại / đặc thù như pin và thiết bị điện tử được định hướng
bổ sung trong các phiên bản dữ liệu và mô hình tiếp theo.

## 4.6. Quy trình suy luận tổng quát

``` text
Ảnh đầu vào
    ↓
Tiền xử lý ảnh
    ↓
Mô hình YOLOv8s
    ↓
Phát hiện đối tượng
    ↓
Lấy class, confidence, bounding box
    ↓
Ánh xạ sang nhóm rác nghiệp vụ
    ↓
Vẽ kết quả lên ảnh
    ↓
Lưu ảnh dự đoán
```

------------------------------------------------------------------------

# TỔNG KẾT NHẬT KÝ PHÁT TRIỂN

Dự án đã hoàn thành các bước chính của quy trình xây dựng một hệ thống
phân loại rác thải sử dụng Deep Learning, bao gồm:

1.  Thiết lập môi trường phát triển local và cloud.
2.  Xây dựng cấu trúc thư mục dự án.
3.  Cài đặt thư viện và môi trường ảo.
4.  Thu thập bộ dữ liệu GARBAGE-CLASSIFICATION-3 từ Roboflow.
5.  Chuẩn hóa dữ liệu theo định dạng YOLO.
6.  Kiểm tra tính toàn vẹn và phân chia dữ liệu.
7.  Thực hiện EDA và trực quan hóa bounding box.
8.  Huấn luyện bốn mô hình YOLO trên GPU Tesla T4.
9.  Đánh giá và đối sánh các mô hình bằng các chỉ số Precision, Recall,
    mAP@50, mAP@50-95 và thời gian inference.
10. Đồng bộ file trọng số và các biểu đồ kết quả về máy local.
11. Xây dựng module suy luận và ánh xạ kết quả từ sáu lớp sang nhóm rác
    nghiệp vụ.

Kết quả thực nghiệm cho thấy YOLOv8s đạt kết quả mAP cao nhất trong nhóm
mô hình được thử nghiệm, trong khi YOLOv8n và YOLO11n có ưu điểm về mức
độ gọn nhẹ và khả năng triển khai trong môi trường hạn chế tài nguyên.

Dự án có thể tiếp tục được mở rộng thông qua:

-   Bổ sung dữ liệu cho lớp `PAPER`, `PLASTIC` và các lớp có hiệu năng
    thấp.
-   Tăng số epoch và thực hiện tuning hyperparameter.
-   Thử nghiệm augmentation phù hợp với điều kiện ảnh thực tế.
-   Bổ sung các lớp rác nguy hại như pin, linh kiện điện tử và hóa chất.
-   Xây dựng giao diện nhận diện trực tiếp từ webcam hoặc camera.
-   Tích hợp mô hình vào ứng dụng web hoặc thiết bị IoT.
-   Đánh giá thêm trên dữ liệu thực tế ngoài tập test ban đầu.
