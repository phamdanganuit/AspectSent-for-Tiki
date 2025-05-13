import pandas as pd
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.constants import CLEANED_REVIEWS_FILE, LABELED_REVIEWS_FILE

# Từ khóa cho các khía cạnh
VAN_CHUYEN_KEYWORDS = [
    'giao hàng', 'ship', 'shipper', 'vận chuyển', 'giao', 'nhận hàng', 'đóng gói',
    'thời gian giao', 'giao nhanh', 'giao chậm', 'phí ship', 'thần tốc', 'gói hàng',
    'vận tải', 'giao dịch', 'đơn vị vận chuyển', 'đơn hàng', 'giao đúng hẹn', 'giao trễ',
    'đóng thùng', 'bọc kĩ', 'bọc cẩn thận', 'bưu điện', 'bưu tá', 'giao hỏng', 'giao thiếu',
    'giao đủ', 'giao đúng', 'giao sai', 'vận chuyển nhanh', 'vận chuyển chậm', 'freeship',
    'phí giao hàng', 'giao tận nơi', 'giao tận nhà', 'giao tận tay', 'giao hàng nhanh chóng'
]

CHAT_LUONG_KEYWORDS = [
    'chất lượng', 'bền', 'hỏng', 'tốt', 'xấu', 'đểu', 'kém', 'chắc chắn',
    'dễ vỡ', 'hư', 'dùng tốt', 'sản phẩm tốt', 'hoạt động', 'dùng', 'chất lượng kém', 'nội dung',
    'hay', 'đẹp', 'xịn', 'độc', 'lạ', 'đẹp trai', 'đẹp gái', 'mới', 'cũ', 
    'hàng chính hãng', 'hàng giả', 'hàng nhái', 'tuyệt vời', 'chất liệu', 'vải', 'da', 'nhựa',
    'kim loại', 'gỗ', 'sắt', 'thép', 'nhôm', 'cao su', 'vải dù', 'vải cotton', 'vải len',
    'vải thô', 'vải mịn', 'bền bỉ', 'dễ hỏng', 'dễ hư', 'dễ bể', 'dễ gãy', 'dễ rách',
    'dễ phai màu', 'phai màu', 'bạc màu', 'ố vàng', 'ố màu', 'đổi màu', 'đẹp mắt',
    'xấu xí', 'thô kệch', 'tinh tế', 'tinh xảo', 'thô sơ', 'chất lượng cao', 'cao cấp',
    'sang trọng', 'đẳng cấp', 'đồ rẻ tiền', 'hàng kém chất lượng', 'hàng tốt', 'hàng xịn',
    'hàng đẹp', 'hàng chất', 'hàng chuẩn', 'hàng đúng mô tả', 'đúng như mô tả', 'như hình',
    'giống hình', 'khác hình', 'không giống hình', 'không như mô tả', 'khác mô tả'
]

GIA_CA_KEYWORDS = [
    'giá', 'đắt', 'rẻ', 'khuyến mãi', 'hợp lý', 'giá cả', 'giá trị', 'tiền',
    'đáng tiền', 'không đáng tiền', 'sale', 'giảm giá', 'rẻ hơn', 'đắt hơn',
    'giá tốt', 'giá cao', 'giá thấp', 'giá hời', 'giá mềm', 'giá phải chăng',
    'giá cạnh tranh', 'giá sinh viên', 'giá ưu đãi', 'ưu đãi', 'chiết khấu',
    'voucher', 'mã giảm giá', 'coupon', 'flash sale', 'deal sốc', 'deal hời',
    'giá gốc', 'giá niêm yết', 'giá thị trường', 'giá bán lẻ', 'giá bán buôn',
    'giá tham khảo', 'giá đề xuất', 'giá thành', 'chi phí', 'phí', 'phụ phí',
    'đắt đỏ', 'rẻ mạt', 'rẻ bèo', 'hời', 'lời', 'lãi', 'lỗ', 'bù lỗ', 'xả kho',
    'xả hàng', 'thanh lý', 'giá thanh lý', 'giá xả kho', 'giá xả hàng'
]

CSKH_KEYWORDS = [
    'nhân viên', 'tư vấn', 'hỗ trợ', 'chăm sóc', 'thái độ', 'phục vụ',
    'nhiệt tình', 'phản hồi', 'chăm sóc khách hàng', 'khách hàng', 'dịch vụ',
    'tư vấn viên', 'người bán', 'shop', 'cửa hàng', 'đại lý', 'nhà phân phối',
    'thái độ phục vụ', 'thái độ nhân viên', 'thái độ shop', 'thái độ người bán',
    'phản hồi nhanh', 'phản hồi chậm', 'trả lời nhanh', 'trả lời chậm',
    'tư vấn nhiệt tình', 'tư vấn tận tâm', 'tư vấn chu đáo', 'tư vấn tốt',
    'tư vấn kém', 'tư vấn sai', 'tư vấn đúng', 'hỗ trợ tốt', 'hỗ trợ kém',
    'hỗ trợ nhiệt tình', 'hỗ trợ tận tâm', 'hỗ trợ chu đáo', 'dịch vụ tốt',
    'dịch vụ kém', 'dịch vụ chất lượng', 'dịch vụ tệ', 'phục vụ tốt', 'phục vụ kém',
    'phục vụ chu đáo', 'phục vụ tận tâm', 'phục vụ nhiệt tình', 'bảo hành',
    'đổi trả', 'hoàn tiền', 'chính sách bảo hành', 'chính sách đổi trả',
    'chính sách hoàn tiền', 'chăm sóc tốt', 'chăm sóc kém', 'chăm sóc chu đáo'
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