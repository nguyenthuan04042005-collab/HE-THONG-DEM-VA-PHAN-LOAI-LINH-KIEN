# Đếm và phân loại linh kiện bằng Canny tự cài đặt

Hệ thống phục vụ bài tập xử lý ảnh: phát hiện biên, tách và đếm **bu-lông,
vòng đệm, đai ốc** trong ảnh nền đơn giản. Thuật toán dùng **NumPy**;
Pillow dùng cho đọc/ghi ảnh, vẽ kết quả; Matplotlib trình bày các bước;
Tkinter cung cấp giao diện desktop. Không gọi hàm xử lý ảnh của OpenCV,
không dùng YOLO, không dùng mô hình học sâu. PyTorch không cần thiết cho baseline này.

Đã có một pipeline chạy được, bộ luật ban đầu và công cụ đánh giá. Độ chính xác
trên ảnh mới vẫn cần được xác nhận bằng tập ảnh chụp độc lập.

## Cấu trúc dự án sau khi hoàn thiện

```text
manual-canny-edge-detector-main/
├── input/                       # Ảnh thật đầu vào hiện có
├── configs/
│   └── default.json              # Tham số Canny, tách vùng và phân loại
├── data/
│   ├── annotations/sample.json   # Nhãn thủ công của ảnh hiện có
│   └── synthetic/                # Ảnh tổng hợp + annotations.json
├── source_code/
│   ├── filters.py                # Kernel đạo hàm Gaussian và tương quan tự viết
│   ├── processing.py             # Gradient, NMS, hysteresis
│   ├── io_utils.py               # Đọc/ghi ảnh và đồ thị hướng gradient
│   ├── canny.py                  # API Canny dùng lại, trả ảnh trung gian
│   ├── config.py                 # Đọc và kiểm tra tham số
│   ├── morphology.py             # Dilation, erosion, closing tự viết
│   ├── segmentation.py           # Flood fill + gán nhãn liên thông tự viết
│   ├── features.py               # PCA, bán kính biên ngoài, lỗ giữa
│   ├── classifier.py             # Luật phân loại + nhãn unknown
│   ├── pipeline.py               # Ghép hệ thống, xuất ảnh/JSON/CSV
│   ├── main.py                   # CLI; giữ chế độ xuất biên cũ
│   ├── demo.py                   # Giao diện desktop
│   └── evaluate.py               # IoU, precision/recall/F1, MAE, confusion matrix
├── scripts/
│   └── generate_synthetic.py     # Tạo dữ liệu tổng hợp có nhãn
├── tests/
│   └── test_system.py            # Kiểm tra thuật toán và pipeline
├── results/
│   ├── demo/<tên_ảnh>/           # Ảnh gắn nhãn, stages/, objects.csv, summary.json
│   └── evaluation/               # Kết quả đánh giá theo từng bộ dữ liệu
├── docs/
│   ├── HUONG_DAN.md              # Giải thích thuật toán và lộ trình thực hành
│   └── ORIGINAL_README.md        # README gốc được giữ để tham khảo
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Chạy trên Windows PowerShell

Chạy các lệnh dưới đây từ thư mục gốc dự án. Môi trường `.venv` hiện có đã
cài đủ thư viện. Dùng đường dẫn Python trong `.venv` để tránh nhầm Python hệ thống.

```powershell
cd D:\XLA\manual-canny-edge-detector-main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe source_code/demo.py --config configs/default.json
```

Nếu chưa có môi trường ảo, tạo bằng `py -3 -m venv .venv` trước khi cài thư viện.

Trong giao diện: **Chọn ảnh → Phân tích → xem từng bước**. Kết quả được lưu
vào `results/demo/<tên_ảnh>/`. Có thể thay sigma, hai ngưỡng, closing và diện tích tối thiểu.

Chạy ảnh hiện có bằng CLI:

```powershell
$imagePath = (Get-ChildItem input -Filter *.jpg | Select-Object -First 1).FullName
.\.venv\Scripts\python.exe source_code/main.py --input "$imagePath" --output results/demo --config configs/default.json
```

Chạy nhiều ảnh cùng phần mở rộng:

```powershell
.\.venv\Scripts\python.exe source_code/main.py --mode parts --input_folder input --input_ext jpg --output results/demo --config configs/default.json
```

Chế độ Canny cũ vẫn chạy được; chế độ này quét các sigma/ngưỡng cố định như mã gốc:

```powershell
.\.venv\Scripts\python.exe source_code/main.py --mode edges --input_folder input --output_folder results/edges --input_ext jpg --output_ext png
```

## Đánh giá và kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe source_code/evaluate.py --annotations data/annotations/sample.json --output results/evaluation/sample --config configs/default.json
.\.venv\Scripts\python.exe scripts/generate_synthetic.py --count 6
.\.venv\Scripts\python.exe source_code/evaluate.py --annotations data/synthetic/annotations.json --output results/evaluation/synthetic --config configs/default.json
```

`sample.json` có bbox xấp xỉ được gán bằng tay, dùng cho phát triển.
Dữ liệu tổng hợp chỉ dùng kiểm tra thuật toán. Không xem hai bộ này là bằng chứng
về độ chính xác trên ảnh chụp mới. Hướng dẫn tạo tập hiệu chỉnh/kiểm thử độc lập,
giải thích đặc trưng và đọc kết quả nằm trong [docs/HUONG_DAN.md](docs/HUONG_DAN.md).

## Giả định và giới hạn

- Linh kiện rời nhau, không chồng lấp, nằm trọn trong ảnh, nền khá đồng nhất.
- Bu-lông nằm để thấy thân dài; vòng đệm/đai ốc thấy lỗ và hình dạng mặt ngoài.
- Đai ốc thuộc dạng có biên ngoài gần lục giác; không bao quát mọi loại đai ốc.
- Biên ngoài cần khép kín. Closing lớn có thể nối nhầm hai linh kiện gần nhau.
- Nhãn `unknown` giữ các trường hợp chưa đủ điều kiện phân loại; vẫn được tính vào tổng.
- Các ngưỡng trong `configs/default.json` là giá trị khởi đầu, cần kiểm tra trên ảnh của bạn.

Phần Canny được phát triển từ mã nguồn có sẵn trong dự án; thông tin gốc
được giữ tại `docs/ORIGINAL_README.md` và `LICENSE`.
