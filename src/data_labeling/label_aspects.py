import pandas as pd
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.constants import CLEANED_REVIEWS_FILE, LABELED_REVIEWS_FILE

# Từ khóa cho các khía cạnh
VAN_CHUYEN_KEYWORDS = [
    'giao hàng', 'ship', 'shipper', 'vận chuyển', 'giao', 'nhận hàng', 'đóng gói',
    'thời gian giao', 'giao nhanh', 'giao chậm', 'phí ship', 'thần tốc', 'gói hàng'
]

CHAT_LUONG_KEYWORDS = [
    'chất lượng', 'bền', 'hỏng', 'tốt', 'xấu', 'đểu', 'kém', 'chắc chắn',
    'dễ vỡ', 'hư', 'dùng tốt', 'sản phẩm tốt', 'hoạt động', 'dùng', 'chất lượng kém', 'nội dung',
    'hay', 'đẹp', 'xịn', 'độc', 'lạ', 'đẹp trai', 'đẹp gái', 'mới', 'cũ', 
    'hàng chính hãng', 'hàng giả', 'hàng nhái', 'tuyệt vời'
]

GIA_CA_KEYWORDS = [
    'giá', 'đắt', 'rẻ', 'khuyến mãi', 'hợp lý', 'giá cả', 'giá trị', 'tiền',
    'đáng tiền', 'không đáng tiền', 'sale', 'giảm giá', 'rẻ hơn', 'đắt hơn'
]

CSKH_KEYWORDS = [
    'nhân viên', 'tư vấn', 'hỗ trợ', 'chăm sóc', 'thái độ', 'phục vụ',
    'nhiệt tình', 'phản hồi', 'chăm sóc khách hàng', 'khách hàng', 'dịch vụ'
]

def load_cleaned_data():
    """
    Đọc dữ liệu đánh giá đã làm sạch từ file CSV.
    
    Returns:
        pandas.DataFrame: DataFrame chứa dữ liệu đánh giá đã làm sạch.
    """
    try:
        df = pd.read_csv(CLEANED_REVIEWS_FILE)
        print(f"Đã đọc {len(df)} đánh giá từ {CLEANED_REVIEWS_FILE}")
        return df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {CLEANED_REVIEWS_FILE}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Lỗi khi đọc file {CLEANED_REVIEWS_FILE}: {e}")
        return pd.DataFrame()

def contains_keyword(text, keywords):
    """
    Kiểm tra xem văn bản có chứa từ khóa nào trong danh sách không.
    
    Args:
        text (str): Chuỗi văn bản cần kiểm tra.
        keywords (list): Danh sách các từ khóa cần tìm.
        
    Returns:
        bool: True nếu văn bản chứa ít nhất một từ khóa, False nếu không.
    """
    if isinstance(text, str):
        text = text.lower()
        for keyword in keywords:
            if keyword in text:
                return True
    return False

def classify_aspect(text):
    """
    Phân loại khía cạnh của đánh giá dựa trên từ khóa.
    
    Args:
        text (str): Chuỗi văn bản cần phân loại.
        
    Returns:
        str: Chuỗi chứa các khía cạnh được phân tách bằng dấu phẩy.
    """
    aspects = []
    
    if contains_keyword(text, VAN_CHUYEN_KEYWORDS):
        aspects.append('ship')
        
    if contains_keyword(text, CHAT_LUONG_KEYWORDS):
        aspects.append('quality')
        
    if contains_keyword(text, GIA_CA_KEYWORDS):
        aspects.append('price')
        
    if contains_keyword(text, CSKH_KEYWORDS):
        aspects.append('cskh')
        
    if not aspects:
        return 'other'
    
    return ', '.join(aspects)

def label_aspects(df):
    """
    Gán nhãn khía cạnh cho các đánh giá.
    
    Args:
        df (pandas.DataFrame): DataFrame chứa dữ liệu đánh giá đã làm sạch.
        
    Returns:
        pandas.DataFrame: DataFrame đã được gán nhãn khía cạnh.
    """
    if df.empty:
        return df
    
    print("Bắt đầu gán nhãn khía cạnh...")
    
    # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
    labeled_df = df.copy()
    
    # Gán nhãn khía cạnh dựa trên cột reviews (thay vì content)
    labeled_df['aspect'] = labeled_df['reviews'].apply(classify_aspect)
    
    # Thống kê phân phối khía cạnh
    aspect_counts = labeled_df['aspect'].value_counts()
    print("\nPhân phối các khía cạnh:")
    print(aspect_counts)
    
    return labeled_df

def save_labeled_data(df):
    """
    Lưu dữ liệu đã gán nhãn vào file CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame chứa dữ liệu đã gán nhãn.
    """
    if df.empty:
        print("Không có dữ liệu để lưu.")
        return
    
    try:
        df.to_csv(LABELED_REVIEWS_FILE, index=False)
        print(f"Đã lưu {len(df)} đánh giá đã gán nhãn vào {LABELED_REVIEWS_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu đã gán nhãn: {e}")

def main():
    """Hàm chính để gán nhãn khía cạnh cho đánh giá."""
    df = load_cleaned_data()
    if not df.empty:
        labeled_df = label_aspects(df)
        save_labeled_data(labeled_df)

if __name__ == "__main__":
    main()