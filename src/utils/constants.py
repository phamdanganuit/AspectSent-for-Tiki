import os

# --- Đường dẫn thư mục ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Thư mục gốc của dự án
DATA_DIR = os.path.join(BASE_DIR, "data")  # Thư mục chứa dữ liệu
BRONZE_DIR = os.path.join(DATA_DIR, "Bronze")  # Dữ liệu thô (chưa xử lý)
SILVER_DIR = os.path.join(DATA_DIR, "Silver")  # Dữ liệu đã xử lý (tách từ, chuẩn hóa)
GOLD_DIR = os.path.join(DATA_DIR, "Gold")  # Dữ liệu sẵn sàng cho fine-tuning PhoBERT
LOGS_DIR = os.path.join(BASE_DIR, "logs")  # Thư mục chứa log
BROWSER_PROFILES_DIR = os.path.join(BASE_DIR, "browser_profiles")  # Thư mục chứa profile trình duyệt

# Đảm bảo các thư mục tồn tại
for directory in [DATA_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, LOGS_DIR, BROWSER_PROFILES_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- BRONZE: Dữ liệu thô chưa qua xử lý ---
CATEGORY_URLS_FILE = os.path.join(BRONZE_DIR, "category_urls.csv")
PRODUCT_URLS_FILE = os.path.join(BRONZE_DIR, "product_urls.csv")
RAW_REVIEWS_FILE = os.path.join(BRONZE_DIR, "raw_reviews.csv")

# --- SILVER: Dữ liệu đã qua xử lý (làm sạch, chuẩn hóa, tách từ) ---
CLEANED_REVIEWS_FILE = os.path.join(SILVER_DIR, "cleaned_reviews.csv")  # Đã làm sạch
NORMALIZED_REVIEWS_FILE = os.path.join(SILVER_DIR, "normalized_reviews.csv")  # Đã chuẩn hóa
TOKENIZED_REVIEWS_FILE = os.path.join(SILVER_DIR, "tokenized_reviews.csv")  # Đã tách từ
LABELED_REVIEWS_FILE = os.path.join(SILVER_DIR, "labeled_reviews.csv")  # Đã gán nhãn

# --- GOLD: Dữ liệu sẵn sàng cho fine-tuning PhoBERT ---
PROCESSED_REVIEWS_FILE = os.path.join(GOLD_DIR, "processed_reviews.csv")  # Dữ liệu cuối cùng
PHOBERT_EMBEDDINGS_FILE = os.path.join(GOLD_DIR, "phobert_embeddings.pkl")  # Embeddings từ PhoBERT
TRAIN_DATA_FILE = os.path.join(GOLD_DIR, "train_data.csv")  # Dữ liệu huấn luyện
TEST_DATA_FILE = os.path.join(GOLD_DIR, "test_data.csv")  # Dữ liệu kiểm thử 