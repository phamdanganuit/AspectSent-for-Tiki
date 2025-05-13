import pandas as pd
import regex as re
import emoji
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path
# Điều chỉnh dòng này nếu cấu trúc thư mục của bạn khác

current_file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_file_dir) if os.path.basename(current_file_dir) in ['features', 'processing'] else current_file_dir # Giả sử file có thể nằm trong features hoặc processing
project_root_dir = os.path.dirname(src_dir) if os.path.basename(src_dir) == 'src' else src_dir
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

# Cố gắng import constants từ vị trí tương đối với cấu trúc dự án
# Ví dụ: nếu file này nằm trong project_root/src/features/
# thì from src.utils.constants sẽ hoạt động nếu project_root được thêm vào sys.path
from src.utils.constants import LABELED_REVIEWS_FILE, NORMALIZED_REVIEWS_FILE


# Từ điển mã teencode
TEENCODE_DICT = {
    'ko': 'không', 'k': 'không', 'kg': 'không', 'khg': 'không', 'hok': 'không', 'hong': 'không', 'hem': 'không', 'hum': 'không', '0': 'không',
    'kh': 'không', 'kô': 'không',
    'dc': 'được', 'dk': 'được', 'dx': 'được', 'đc': 'được',
    'vs': 'với', 'wa': 'quá', 'wá': 'quá',
    'j': 'gì', 'z': 'gì', 'ji': 'gì',
    'ok': 'tốt', 'okie': 'tốt', 'oke': 'tốt', 'okê': 'tốt', 'oki': 'tốt', 'okla': 'tốt',
    'thanks': 'cảm ơn', 'thank': 'cảm ơn', 'tks': 'cảm ơn',
    'fb': 'facebook', 'insta': 'instagram', 'ins': 'instagram',
    'rep': 'trả lời', 'reply': 'trả lời', 'ib': 'nhắn tin',
    'mn': 'mọi người', 'mng': 'mọi người',
    'ntn': 'như thế nào',
    'v': 'vậy', 'zậy': 'vậy', 'za': 'vậy',
    'mk': 'mình', 'm': 'mình',
    'b': 'bạn',
    'r': 'rồi', 'rr': 'rồi',
    'cx': 'cũng', 'cxung': 'cũng',
    'sz': 'kích thước', 'size': 'kích thước',
    'lun': 'luôn',
    'h': 'giờ',
    'đt': 'điện thoại', 'dt': 'điện thoại',
    'sp': 'sản phẩm', 'spham': 'sản phẩm',
    'ship': 'giao hàng', 'sip': 'giao hàng', 'shiper': 'người giao hàng', 'shipper': 'người giao hàng',
    'nv': 'nhân viên',
    'đg': 'đóng gói', 'donggoi': 'đóng gói',
    'sd': 'sử dụng',
    'bh': 'bảo hành',
    'auth': 'chính hãng', 'authentic': 'chính hãng', 'real': 'thật',
    'fake': 'giả',
    'check': 'kiểm tra',
    'ord': 'đặt hàng', 'order': 'đặt hàng',
    'feedback': 'phản hồi', 'review': 'đánh giá', 'rv': 'đánh giá',
    'bt': 'bình thường', # 'bt' cũng có thể là 'biết'. Cần quyết định.
    'nh': 'nhanh',
    'chậm': 'chậm', 'chậm trễ': 'chậm', 'trễ': 'chậm',
}

# Từ điển viết tắt
ABBREVIATION_DICT = {
    'tp': 'thành phố',
    'hcm': 'hồ chí minh', 'sg': 'sài gòn',
    'hn': 'hà nội', 'đn': 'đà nẵng',
    'cty': 'công ty',
    'bs': 'bác sĩ',
    'bv': 'bệnh viện',
    'cs': 'chất liệu', 
    'ch': 'cấu hình', # 'ch' cũng có thể là 'cửa hàng' trong TEENCODE_DICT
    'mnc': 'màn hình cong', 
    'hsd': 'hạn sử dụng',
    'hdsd': 'hướng dẫn sử dụng',
    'hd': 'hướng dẫn',
    'shop': 'cửa hàng' # Thêm shop vào đây để có độ ưu tiên
}


def load_cleaned_data():
    try:
        df = pd.read_csv(LABELED_REVIEWS_FILE)
        print(f"Đã đọc {len(df)} đánh giá từ {LABELED_REVIEWS_FILE}")
        return df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {LABELED_REVIEWS_FILE}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Lỗi khi đọc file {LABELED_REVIEWS_FILE}: {e}")
        return pd.DataFrame()

def normalize_text(text):
    if not isinstance(text, str) or pd.isna(text) or text == "N/A":
        return ""
    
    text = str(text).lower()
    text = re.sub(r'\*', ' ', text)
    text = re.sub(r'\b(xem thêm|thu gọn)\b', ' ', text, flags=re.IGNORECASE)
    
    words = text.split()
    processed_words = []
    temp_teencode_dict = {k: v for k, v in TEENCODE_DICT.items()} # Tạo bản sao để có thể sửa đổi
    if 'bt' in temp_teencode_dict and 'bt' in ABBREVIATION_DICT: # Ví dụ xử lý xung đột
        del temp_teencode_dict['bt'] # Loại bỏ 'bt' từ teencode nếu nó có trong abbreviation

    for word in words:
        if word in ABBREVIATION_DICT:
            processed_words.append(ABBREVIATION_DICT[word])
        elif word in temp_teencode_dict:
            processed_words.append(temp_teencode_dict[word])
        else:
            processed_words.append(word)
    text = ' '.join(processed_words)
    
    text = emoji.demojize(text, delimiters=(" EMOJI_", " "))
    text = re.sub(r'([\p{L}\p{N}])\1\1+', r'\1\1', text, flags=re.UNICODE) 
    text = re.sub(r'[^\p{L}\p{N}\s.,!?]', ' ', text, flags=re.UNICODE)
    
    # Loại bỏ phần tách từ bằng ViTokenizer trong normalize_text
    # Giờ đây quá trình này sẽ được thực hiện trong tokenize_text.py
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Định nghĩa hàm để chuyển đổi chuỗi aspect thành danh sách các mã code
def encode_aspect_to_code_list(aspect_string, mapping_dict):
    """
    Chuyển đổi một chuỗi các aspect (phân tách bằng dấu phẩy) 
    thành một danh sách các mã số duy nhất, đã sắp xếp.
    Nếu có aspect lạ không có trong mapping, mã của 'khác' sẽ được thêm vào.
    """
    if pd.isna(aspect_string) or not isinstance(aspect_string, str) or not aspect_string.strip():
        return [] # Trả về danh sách rỗng cho đầu vào không hợp lệ hoặc rỗng

    codes = set() # Sử dụng set để đảm bảo mã không bị trùng lặp
    individual_aspects_raw = aspect_string.split(',')
    found_unmappable_term = False

    for asp_raw in individual_aspects_raw:
        cleaned_asp = asp_raw.strip().lower()
        if not cleaned_asp: # Bỏ qua nếu sau khi strip, chuỗi trở thành rỗng
            continue
            
        if cleaned_asp in mapping_dict:
            codes.add(mapping_dict[cleaned_asp])
        else:
            # print(f"Debug: Term '{cleaned_asp}' không có trong mapping.") # Dòng debug (có thể xóa)
            found_unmappable_term = True
            
    # Nếu tìm thấy bất kỳ term nào không map được, và 'khác' có trong mapping, thêm mã của 'khác'
    if found_unmappable_term and 'khác' in mapping_dict:
        codes.add(mapping_dict['khác'])
        
    return sorted(list(codes)) # Trả về danh sách các mã đã sắp xếp

def normalize_data(df):
    if df.empty:
        print("DataFrame đầu vào rỗng.")
        return df
    
    print("Bắt đầu chuẩn hóa dữ liệu...")
    normalized_df = df.copy()
    
    if 'reviews' in normalized_df.columns:
        normalized_df['normalized_text'] = normalized_df['reviews'].apply(normalize_text)
        normalized_df.dropna(subset=['normalized_text'], inplace=True)
        normalized_df = normalized_df[normalized_df['normalized_text'].str.strip().str.len() > 0]
    else:
        print("Cảnh báo: Không tìm thấy cột 'reviews'.")

    print(f"Số lượng dòng sau khi chuẩn hóa text review và loại bỏ review rỗng: {len(normalized_df)}")
    
    if 'sentiment' in normalized_df.columns:
        valid_ratings = ['Cực kì hài lòng', 'Hài lòng', 'Không hài lòng', 'Rất không hài lòng', 'Bình thường']
        rows_before_sentiment_filter = len(normalized_df)
        normalized_df.dropna(subset=['sentiment'], inplace=True)
        normalized_df = normalized_df[normalized_df['sentiment'].isin(valid_ratings)]
        rows_after_sentiment_filter = len(normalized_df)
        print(f"Đã loại bỏ {rows_before_sentiment_filter - rows_after_sentiment_filter} dòng có giá trị sentiment không hợp lệ hoặc NaN.")
        
        if not normalized_df.empty:
            rating_mapping = {
                'Rất không hài lòng': 1, 'Không hài lòng': 2, 'Bình thường': 3, 
                'Hài lòng': 4, 'Cực kì hài lòng': 5
            }
            normalized_df['sentiment'] = normalized_df['sentiment'].map(rating_mapping).astype('category')
    else:
        print("Cảnh báo: Không tìm thấy cột 'sentiment'.")

    if 'type_product' in normalized_df.columns:
        normalized_df['type_product'] = normalized_df['type_product'].astype('category')
    else:
        print("Cảnh báo: Không tìm thấy cột 'type_product'.")
        
    # ---- THAY ĐỔI CÁCH XỬ LÝ CỘT 'ASPECT' ----
    if 'aspect' in normalized_df.columns:
        print("Bắt đầu encoding cột 'aspect' thành danh sách các mã số...")
        aspect_code_mapping = { # Đổi tên để rõ ràng hơn
            'ship': 4, 
            'quality': 2,
            'price': 3,
            'cskh': 1,
            'other': 0 
        }
        
    
        normalized_df['aspect'] = normalized_df['aspect'].apply(
            lambda x: encode_aspect_to_code_list(x, aspect_code_mapping)
        )
        
        print("Đã hoàn thành encoding cột 'aspect'. Cột mới 'aspect' đã được tạo chứa danh sách mã.")
     
    else:
        print("Cảnh báo: Không tìm thấy cột 'aspect' để encoding.")
    # ---- KẾT THÚC THAY ĐỔI ----
        
    return normalized_df

def save_normalized_data(df, filepath):
    if df.empty:
        print("DataFrame rỗng, không có dữ liệu để lưu.")
        return
    try:
        df.to_csv(filepath, index=False)
        print(f"Đã lưu {len(df)} đánh giá đã chuẩn hóa vào {filepath}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu đã chuẩn hóa: {e}")

def main():
    df = load_cleaned_data()
    if not df.empty:
        normalized_df = normalize_data(df)
        if not normalized_df.empty:
            # In ra vài dòng của cột aspect_codes để kiểm tra
            if 'aspect' in normalized_df.columns:
                print("\nXem trước cột 'aspect' sau khi xử lý:")
                print(normalized_df[['aspect']].head(10))
            save_normalized_data(normalized_df, NORMALIZED_REVIEWS_FILE)
        else:
            print("DataFrame rỗng sau khi chuẩn hóa, không có gì để lưu.")
    else:
        print("Không tải được dữ liệu ban đầu.")

if __name__ == "__main__":
    main()