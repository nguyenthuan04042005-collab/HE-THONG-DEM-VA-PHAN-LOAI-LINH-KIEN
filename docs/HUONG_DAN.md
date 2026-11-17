# Hướng dẫn hoàn thiện và trình bày hệ thống

## 1. Kiến trúc và vai trò của Canny

```text
RGB → ảnh xám → đạo hàm Gaussian → độ lớn/hướng gradient
    → NMS → ngưỡng kép + hysteresis → ảnh biên
    → closing → flood fill nền → vùng bao ngoài
    → gán nhãn từng vùng → đặc trưng → bộ luật → đếm + ảnh gắn nhãn
```

Canny quyết định vị trí đường biên dùng để tạo vùng vật thể. Không có nhánh
phân ngưỡng toàn ảnh để thay thế Canny. Cường độ xám chỉ hỗ trợ xác nhận
các vùng bên trong có giống nền hay không khi tìm lỗ.

Các thuật toán dùng NumPy và cấu trúc dữ liệu Python. `np.linalg.eigh`
chỉ giải bài toán trị riêng của ma trận 2 × 2, không phải hàm nhận dạng ảnh.
Pillow đọc/ghi ảnh và vẽ; Matplotlib tạo hình tổng quan; Tkinter tạo giao diện.
Không import `cv2`, YOLO, SciPy hoặc scikit-image trong phần xử lý.

Các sửa đổi với Canny gốc:

- Thống nhất `X` theo cột, `Y` theo hàng, sửa cặp láng giềng chéo của NMS.
- Padding lặp giá trị mép để tránh biên giả do đệm bằng 0.
- Tương quan giữ quy ước dấu của kernel cũ. Hai đạo hàm cùng đảo dấu so với
  tích chập toán học, không đổi hướng biên modulo 180°.
- Lặp qua các phần tử kernel và xử lý cả mảng ảnh để giảm chi phí vòng lặp.
- Hysteresis đánh dấu khi đưa pixel vào stack, tránh đưa lặp nhiều lần.
- Hai ngưỡng phải thỏa `0 < low <= high <= 255`.

Đạo hàm Gaussian kết hợp làm trơn và tính đạo hàm, không cần áp thêm Sobel.
Các ngưỡng hiện dùng trên ảnh NMS chuẩn hóa về 0–255 theo từng ảnh,
không phải đơn vị độ lớn gradient gốc. Một ảnh tương phản thấp vẫn có thể
sinh biên; cần xem ảnh biên trước khi tin kết quả đếm.

## 2. Thực hành tách từng linh kiện

Đọc `morphology.py` và `segmentation.py` trước khi đọc bộ phân loại.

1. Nhận ảnh biên 0/255 từ Canny, đổi sang mask boolean.
2. Dilation bằng lân cận 3 × 3, sau đó erosion để tạo closing.
3. Flood fill nền từ bốn mép ảnh theo liên thông 4.
4. Đảo mask nền ngoài để nhận các vùng bị đường biên bao kín, gồm cả pixel biên.
5. Gán nhãn liên thông 8 bằng BFS để mỗi vùng bao ngoài nhận một ID.
6. Loại vùng quá nhỏ, quá lớn, chạm mép ảnh (cách mép dưới 3 px, vì Canny luôn
   bỏ viền 1 px) hoặc chỉ là sợi biên hở (chiếm dưới 8% khung bao của nó).
   Linh kiện bị mép ảnh cắt mất một phần vì vậy không được đếm.

Flood fill dùng các đoạn liên tiếp trên hàng để giảm số phần tử phải duyệt.
Gán nhãn dùng BFS tường minh để bạn dễ giải thích và kiểm tra.

Vòng đệm có biên ngoài và biên lỗ. Sau lấp đầy, nó phải trở thành **một vùng
bao ngoài**, không phải hai đối tượng. Không đếm trực tiếp các đường biên.

Kiểm tra `stages/edges.png`, `stages/closed.png`, `stages/labels.png`.
Mỗi linh kiện phải có đúng một vùng màu. Biên chưa kín có thể để nền tràn
vào vật thể, làm mất đối tượng; mảnh biên lớn cũng có thể bị đếm nhầm.
Hệ thống chưa tự sửa mọi trường hợp biên hỏng.

Nếu sai bước này:

- Mất vật thể: xem biên có kín không; giảm ngưỡng nếu biên bị mất.
- Nhiều nhiễu: tăng sigma hoặc ngưỡng, kiểm tra lại chi tiết nhỏ.
- Vùng nhỏ xuất hiện: tăng `min_area` sau khi xem kích thước linh kiện thật.
- Hai vật thể thành một vùng: giảm closing. Vật thể thật sự chạm nhau nằm
  ngoài giả định của baseline, cần bài toán tách vật thể chạm nhau riêng.

`min_area` tính theo pixel, cần hiệu chỉnh khi độ phân giải thay đổi.

## 3. Thực hành trích đặc trưng

Đọc `features.py`. Bbox dùng `[x0, y0, x1, y1]`, cận trên không bao gồm.
Tọa độ trong báo cáo dùng `(x, y)`; mảng NumPy truy cập `[y, x]`.

### Diện tích và tâm

- `outer_area`: số pixel của vùng bao ngoài đã lấp đầy.
- `centroid`: trung bình tọa độ pixel trong vùng bao ngoài.
- `material_area_estimate`: diện tích bao ngoài trừ diện tích lỗ ước lượng.

### Độ kéo dài bằng PCA

Từ tọa độ pixel đã trừ tâm, tính:

```text
covariance = centered.T @ centered / number_of_pixels
elongation = sqrt(largest_eigenvalue / smallest_eigenvalue)
```

Giá trị gần 1 cho hình gần đều; giá trị lớn cho vật thể dài. Đặc trưng này
ít phụ thuộc góc xoay trong mặt phẳng hơn bbox ngang/dọc, không khôi phục
vật thể bị che hoặc xoay trong không gian 3D.

### Biên ngoài và đặc trưng lục giác

Lấy biên vùng bao ngoài bằng `mask & ~erode(mask)`. Biên này không chứa
biên lỗ vì mask đã lấp đầy; không cần hàm tìm contour của OpenCV.

Chuẩn hóa tọa độ theo PCA để giảm méo elip, chia góc thành 180 khoảng,
lấy bán kính ngoài cùng trong mỗi khoảng và nội suy các khoảng thiếu.

- `radial_cv`: độ lệch chuẩn của bán kính sau khi chuẩn hóa trung bình về 1.
  Hình tròn có giá trị nhỏ; lục giác có biến thiên lớn hơn.
- `hex_score`: biên độ điều hòa bậc 6 của dãy bán kính chuẩn hóa:

```text
hex_score = 2 * abs(mean((radius_normalized - 1) * exp(-6j * theta)))
```

Đây là đặc trưng hình học cho đai ốc gần lục giác, không phải xác suất dự đoán.
Không gọi nó là confidence hay cam kết đúng với mọi vật thể có sáu phần lồi/lõm.

`outer_perimeter_grid` đếm các cạnh pixel tiếp xúc nền theo lưới 4 hướng.
`circularity_grid = 4*pi*area/perimeter_grid**2` chỉ dùng chẩn đoán: chu vi
lưới có sai lệch và phụ thuộc hướng, vòng tròn không nhất thiết cho giá trị
gần 1. Bộ luật không dùng circularity này để quyết định loại linh kiện.

### Lỗ giữa

Giữ ảnh biên đã closing. Bên trong mỗi vùng bao ngoài:

1. Lấy các vùng không phải pixel biên, gán nhãn liên thông 4.
2. Chọn vùng có diện tích hợp lý so với vùng bao ngoài và ở gần tâm.
3. So sánh trung vị mức xám với nền quanh bbox; giữ vùng đủ giống nền.

Điều này giúp loại vùng vật liệu giữa các biên do phản sáng, nhưng vẫn là
heuristic cho nền đơn giản. Phản sáng giống nền hoặc biên lỗ hở có thể gây nhầm.
`hole_count` và diện tích lỗ là giá trị ước lượng.

Kiểm tra `stages/holes.png`: nền đen, lỗ được chọn màu trắng.

## 4. Thực hành phân loại và đếm

Đọc `classifier.py` theo thứ tự luật:

```text
elongation >= bolt_min_elongation
    → bolt
elongation > round_max_elongation
    → unknown
không tìm được đúng một lỗ giữa
    → unknown
hex_score >= nut_min_hex_score và radial_cv <= nut_max_radial_cv
    → nut
hex_score < nut_min_hex_score và radial_cv <= washer_max_radial_cv
    → washer
còn lại
    → unknown
```

Luật bu-lông giả định đầu vào chỉ có ba loại của đề, không phải bộ phát hiện
bu-lông tổng quát giữa mọi vật thể dài. Hai loại còn lại dùng **hình dạng biên
ngoài kết hợp lỗ giữa**, không chỉ có/không có lỗ.

`pipeline.py` tăng bộ đếm một lần cho mỗi ID. `total` gồm cả `unknown`;
tổng ba loại đã biết có thể nhỏ hơn `total`.

Mở `objects.csv`, so sánh đặc trưng giữa các loại và xem cột `reason`.
Hiệu chỉnh bằng nhiều mẫu, không chọn ngưỡng chỉ để một ảnh cho kết quả đúng.

## 5. Demo và các file đầu ra

Chạy `source_code/demo.py` bằng Python trong `.venv`. Tkinter là giao diện
desktop, không cần web server hoặc tài khoản bên ngoài.

1. Chọn ảnh trong `input`.
2. Bấm Phân tích; tác vụ chạy ở luồng nền để cửa sổ vẫn phản hồi.
3. Kiểm tra edges → closed → labels → holes → overlay.
4. Đối chiếu bảng đặc trưng và số lượng hiển thị.
5. Thay tham số, phân tích lại, ghi lại lý do thay đổi.

Kết quả tại `results/demo/<tên_ảnh>/`:

- `overlay.png`: ảnh gắn bbox, ID, tên loại.
- `overview.png`: hình tổng quan sáu bước.
- `stages/`: ảnh xám, gradient, NMS, biên, vùng vật thể, lỗ.
- `objects.csv`: một dòng mỗi vật thể.
- `summary.json`: tham số, đặc trưng, nhãn, số lượng, cảnh báo, thời gian.

Chạy lại cùng tên ảnh trong cùng output sẽ cập nhật kết quả cũ. Muốn so sánh
cấu hình, dùng các thư mục `--output` khác nhau. Thời gian trong JSON chỉ
tính pipeline xử lý, chưa gồm vẽ và ghi file.

## 6. Tạo nhãn và đánh giá

Tạo JSON như sau; đường dẫn ảnh tính từ nơi đặt JSON:

```json
{
  "images": [
    {
      "image": "../../input/ten_anh.jpg",
      "objects": [
        {"label": "bolt", "bbox": [20, 30, 180, 260]},
        {"label": "washer", "bbox": [220, 50, 320, 150]},
        {"label": "nut", "bbox": [400, 50, 510, 170]}
      ]
    }
  ]
}
```

Tọa độ trên chỉ minh họa định dạng, không phải nhãn ảnh có sẵn. Dùng lớp
`bolt`, `washer`, `nut`; bbox bao toàn vật thể, gồm cả đầu bu-lông.
Ảnh không có linh kiện dùng `"objects": []` để kiểm tra đếm nhầm.

Tách dữ liệu trước khi chọn ngưỡng:

- Tập hiệu chỉnh: xem đặc trưng và chọn tham số.
- Tập kiểm thử: ảnh chụp khác, chỉ chạy sau khi chốt tham số.
- Tránh để cùng một ảnh hoặc các bản xoay/cắt của nó ở cả hai tập.
- Thử nhiều góc xoay, kích thước, vị trí, số lượng trong điều kiện lý tưởng
  đã công bố. Có thể bắt đầu với 10–20 ảnh thật và mở rộng khi phát hiện lỗi.

Sau khi gán nhãn, chạy:

```powershell
.\.venv\Scripts\python.exe source_code/evaluate.py --annotations data/annotations/test.json --config configs/default.json --output results/evaluation/test
```

Ghép bbox theo IoU giảm dần, một-một, không xét nhãn khi ghép. Ngưỡng mặc
định 0.5, thay bằng `--iou`. Đây là cách greedy đơn giản cho ảnh vật thể
tách rời, không phải bộ đánh giá mAP chuẩn COCO.

Các chỉ số trong `metrics.json`:

- `detection_precision`, `detection_recall`: tách/định vị đúng, chưa xét lớp.
- `classification_accuracy_on_matched`: đúng lớp trong các cặp đã ghép.
  Đọc cùng recall vì bỏ sót vật thể có thể làm accuracy này đẹp hơn thực tế.
- `class_metrics`: TP, FP, FN, precision, recall, F1 và support từng loại.
- `count_mae_per_class`: trung bình sai số tuyệt đối số lượng từng loại mỗi ảnh.
- `exact_count_image_accuracy`: tỷ lệ ảnh có toàn bộ số đếm đúng, kể cả unknown bằng 0.

Nếu mẫu số bằng 0, tỷ lệ được ghi 0; đọc support và số vật thể để phân biệt
không có dữ liệu với kết quả kém. MAE bằng 0 cũng chưa bảo đảm đúng từng
vật thể nếu các lỗi bù trừ nhau.

`confusion_matrix.csv`: hàng là lớp thật, cột là dự đoán; `missed` là bỏ sót,
`spurious` là dự đoán dư. `counts.csv` cho số thật/dự đoán/sai số từng ảnh.
Ảnh gắn nhãn từng mẫu nằm trong `images/` của lần đánh giá.

`data/annotations/sample.json` chứa bbox xấp xỉ gán tay của ảnh hiện có. Đây
là ví dụ phát triển đã xem khi xây hệ thống, không phải kiểm thử độc lập.
`scripts/generate_synthetic.py` tạo hình đơn giản có nhãn để tìm lỗi thuật toán;
kết quả tốt trên dữ liệu này không suy ra độ chính xác ảnh thật.

## 7. Phần bạn cần tiếp tục làm

Mã baseline đã có đủ các khối. Tiếp tục theo các mốc có thể kiểm chứng:

1. Giải thích và chạy kiểm tra hướng gradient/NMS trong `tests/test_system.py`.
2. Xác nhận mỗi linh kiện ứng với một vùng trong `labels.png`.
3. Đọc `objects.csv`, liên hệ từng số với hình dạng trong ảnh.
4. Thu thập ảnh thật, gán bbox và nhãn; tách hiệu chỉnh/kiểm thử.
5. Hiệu chỉnh trên tập hiệu chỉnh, ghi cấu hình đã chốt thành JSON.
6. Đánh giá tập kiểm thử, phân tích ảnh sai; không sửa nhãn để khớp dự đoán.
7. Chuẩn bị demo và báo cáo: giả định, công thức, ảnh trung gian, kết quả,
   lỗi còn tồn tại, phạm vi áp dụng.

Ghi rõ phần Canny kế thừa và phần sửa/bổ sung. Không tuyên bố hệ thống xử lý
linh kiện chồng lấp hoặc đạt độ chính xác tổng quát khi chưa kiểm chứng.
