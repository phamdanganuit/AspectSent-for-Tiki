# Ứng dụng Phân tích Đánh giá Sản phẩm Tiki

Ứng dụng web này giúp phân tích cảm xúc và khía cạnh từ các đánh giá sản phẩm trên Tiki.vn, sau đó đưa ra khuyến nghị có nên mua sản phẩm hay không.

## Tính năng

- Thu thập thông tin sản phẩm và đánh giá từ URL Tiki
- Tiền xử lý dữ liệu đánh giá
- Phân loại cảm xúc và khía cạnh bằng mô hình PhoBERT đã fine-tune
- Hiển thị kết quả dưới dạng biểu đồ trực quan
- Đưa ra khuyến nghị dựa trên phân tích
- Hiển thị các đánh giá tiêu biểu

## Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch 1.7+ (với CUDA tùy chọn)
- Flask
- Các thư viện trong requirements.txt

### Các bước cài đặt

1. Cài đặt các gói phụ thuộc:

```bash
pip install -r requirements.txt
```

2. Đảm bảo đã fine-tune mô hình PhoBERT (theo hướng dẫn trong thư mục gốc của dự án).

3. Khởi động ứng dụng:

```bash
python app/app.py
```

4. Mở trình duyệt và truy cập: http://localhost:5000

## Sử dụng

1. Truy cập giao diện web qua trình duyệt
2. Nhập URL của sản phẩm trên Tiki.vn vào ô tìm kiếm
3. Nhấn nút "Phân tích"
4. Đợi quá trình phân tích hoàn tất
5. Xem kết quả phân tích và khuyến nghị

## Cấu trúc thư mục

```
app/
├── app.py              # File chính của ứng dụng Flask
├── static/             # Thư mục chứa CSS, JS, hình ảnh
│   └── style.css       # CSS cho ứng dụng
├── templates/          # Thư mục chứa các template HTML
│   └── index.html      # Giao diện chính
└── README.md           # Hướng dẫn này
```

## Lưu ý

- Ứng dụng cần kết nối internet để thu thập dữ liệu từ Tiki.vn
- Quá trình phân tích cần GPU để tối ưu tốc độ (không bắt buộc)
- Phiên bản demo giới hạn ở 3 trang đánh giá đầu tiên để tăng tốc độ 