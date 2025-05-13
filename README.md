# Dự án Phân tích Cảm xúc Đánh giá Tiki

Dự án này thu thập và phân tích dữ liệu đánh giá sản phẩm từ trang Tiki.vn. Dự án gồm các giai đoạn:
1. Thu thập dữ liệu (scraping)
2. Tiền xử lý và làm sạch dữ liệu
3. Chuẩn hóa văn bản
4. Gán nhãn khía cạnh

## Cấu trúc dự án

```
.
├── browser_profiles/     # Thư mục chứa profiles trình duyệt
├── data/                 # Thư mục chứa dữ liệu
│   ├── Bronze/           # Dữ liệu thô
│   ├── Silver/           # Dữ liệu đã tiền xử lý
│   └── Gold/             # Dữ liệu đã gán nhãn và sẵn sàng sử dụng
├── logs/                 # Thư mục chứa logs
├── src/                  # Mã nguồn
│   ├── __init__.py       
│   ├── pipeline.py       # File điều phối toàn bộ pipeline
│   ├── data_collection/  # Thu thập dữ liệu
│   │   ├── __init__.py
│   │   ├── crawl_category_urls.py
│   │   ├── crawl_product_urls.py
│   │   └── crawl_product_reviews.py
│   ├── data_preprocessing/  # Tiền xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── clean_data.py
│   │   └── normalize_text.py
│   ├── data_labeling/    # Gán nhãn dữ liệu
│   │   ├── __init__.py
│   │   └── label_aspects.py
│   └── utils/            # Tiện ích và hàm hỗ trợ
│       ├── __init__.py
│       ├── constants.py
│       └── driver_setup.py
├── README.md
└── requirements.txt
```

## Cài đặt

1. Clone dự án:
```bash
git clone <repository-url>
cd <repository-dir>
```

2. Tạo môi trường ảo và cài đặt thư viện:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

3. Đảm bảo đã cài đặt Chrome và ChromeDriver phù hợp với phiên bản Chrome của bạn.

## Sử dụng

### Chạy toàn bộ pipeline:
```bash
python src/pipeline.py
```

### Chạy một hoặc nhiều bước cụ thể:
```bash
python src/pipeline.py --steps crawl_category crawl_products
```

Các bước khả dụng:
- `crawl_category`: Thu thập URL danh mục sản phẩm
- `crawl_products`: Thu thập URL sản phẩm
- `crawl_reviews`: Thu thập đánh giá sản phẩm
- `clean_data`: Làm sạch dữ liệu
- `normalize_text`: Chuẩn hóa văn bản
- `label_aspects`: Gán nhãn khía cạnh cho đánh giá

## Yêu cầu hệ thống

- Python 3.8+
- Chrome Browser
- 4GB RAM trở lên (khuyến nghị)
- Kết nối internet ổn định

## Ghi chú

- Tất cả dữ liệu thu thập được lưu trong thư mục `data/`
- Logs được lưu trong thư mục `logs/`
- Profiles trình duyệt được lưu trong thư mục `browser_profiles/` 