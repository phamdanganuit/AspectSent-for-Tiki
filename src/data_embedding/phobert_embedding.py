import pandas as pd
import torch
import ast  # Để chuyển đổi chuỗi "[1, 2]" thành list [1, 2] một cách an toàn
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.constants import NORMALIZED_REVIEWS_FILE, GOLD_DIR

# --- Cấu hình CỐ ĐỊNH ---
os.makedirs(GOLD_DIR, exist_ok=True)

INPUT_IDS_FILE = os.path.join(GOLD_DIR, "input_ids.pt")
ATTENTION_MASKS_FILE = os.path.join(GOLD_DIR, "attention_masks.pt")
SENTIMENT_LABELS_FILE = os.path.join(GOLD_DIR, "sentiment_labels.pt")
ASPECT_LABELS_FILE = os.path.join(GOLD_DIR, "aspect_labels.pt")
FINETUNING_METADATA_FILE = os.path.join(GOLD_DIR, "finetuning_metadata.csv")

PHOBERT_MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 256

ASPECT_MAPPING = {
    'other': 0,
    'cskh': 1,
    'quality': 2,
    'price': 3,
    'ship': 4
}
NUM_TOTAL_ASPECTS = len(ASPECT_MAPPING)
SENTIMENT_ADJUSTMENT = -1




# --- Các hàm trợ giúp ---
def parse_aspect_string_to_list(aspect_str: str):
    if pd.isna(aspect_str) or not isinstance(aspect_str, str) or not aspect_str.strip():
        return []
    try:
        parsed_list = ast.literal_eval(aspect_str)
        if isinstance(parsed_list, list):
            return [int(item) for item in parsed_list if isinstance(item, (int, float))]
        return []
    except (ValueError, SyntaxError, TypeError):
        return []

def create_multi_hot_vector(codes_list: list, num_total_labels: int):
    multi_hot = torch.zeros(num_total_labels, dtype=torch.float)
    if codes_list:
        valid_codes = [code for code in codes_list if 0 <= code < num_total_labels]
        if valid_codes:
            multi_hot[valid_codes] = 1.0
    return multi_hot

# --- Hàm chính để xử lý dữ liệu ---
def prepare_data_for_finetuning(df_path: str, tokenizer_name: str, max_seq_length: int,
                                aspect_map_dict: dict, num_aspects: int, sentiment_adj: int):
    # 1. Tải dữ liệu
    try:
        df = pd.read_csv(df_path)
        print(f"Đã đọc {len(df)} dòng từ {df_path}")
    except Exception as e:
        print(f"Lỗi khi đọc file {df_path}: {e}")
        return None, None, None, None, None

    if df.empty:
        print("DataFrame rỗng.")
        return None, None, None, None, None

    required_columns = ['reviews', 'sentiment', 'aspect']
    for col in required_columns:
        if col not in df.columns:
            print(f"Lỗi: Thiếu cột '{col}' trong file đầu vào.")
            return None, None, None, None, None
            
    # Xử lý NaN trong các cột quan trọng trước khi làm gì other
    df.dropna(subset=['reviews', 'sentiment'], inplace=True) # Loại bỏ dòng có reviews hoặc sentiment NaN
    df['reviews'] = df['reviews'].astype(str) # Đảm bảo reviews là string
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce') # Chuyển sentiment sang số, lỗi thì thành NaN
    df.dropna(subset=['sentiment'], inplace=True) # Loại bỏ dòng sentiment không phải số
    df['sentiment'] = df['sentiment'].astype(int) # Chuyển sentiment thành int

    if df.empty:
        print("DataFrame trở nên rỗng sau khi loại bỏ NaN ở reviews/sentiment.")
        return None, None, None, None, None

    print(f"Số dòng sau khi loại bỏ NaN ở reviews/sentiment: {len(df)}")

    # 2. Xử lý cột 'aspect' thành list các mã số
    print("Đang xử lý cột 'aspect' thành danh sách mã số...")
    # Hàm lambda này sẽ áp dụng logic parse_aspect_string_to_list và xử lý unmapped terms
    def process_aspect_string(aspect_str, mapping_dict_for_lambda):
        if pd.isna(aspect_str) or not isinstance(aspect_str, str) or not aspect_str.strip():
            return []
        
        codes = set()
        individual_aspects_raw = aspect_str.split(',')
        found_unmappable_term_in_row = False

        for asp_raw in individual_aspects_raw:
            cleaned_asp = asp_raw.strip().lower()
            if not cleaned_asp: continue # Bỏ qua term rỗng sau khi strip
            
            # Thử parse nếu cleaned_asp là số dạng string (ví dụ "2" thay vì "[2]")
            # Hoặc nếu nó là key dạng text trong mapping
            try: # Thử coi nó là số trước
                code_val = int(cleaned_asp)
                if 0 <= code_val < num_aspects: 
                     pass 
            except ValueError: 
                pass
                     
        try:
            parsed_list_from_string = ast.literal_eval(aspect_str)
            if isinstance(parsed_list_from_string, list):
                for item_code in parsed_list_from_string:
                    if isinstance(item_code, int) and 0 <= item_code < num_aspects:
                        codes.add(item_code)
                    else: # item_code không hợp lệ hoặc không phải số
                        found_unmappable_term_in_row = True
            else: # ast.literal_eval không trả về list
                found_unmappable_term_in_row = True
        except (ValueError, SyntaxError, TypeError):
            individual_terms_from_split = aspect_str.split(',')
            explicitly_parsed_terms = False
            for term_raw in individual_terms_from_split:
                cleaned_term = term_raw.strip().lower()
                if not cleaned_term: continue
                if cleaned_term in mapping_dict_for_lambda:
                    codes.add(mapping_dict_for_lambda[cleaned_term])
                    explicitly_parsed_terms = True
                else:
                    found_unmappable_term_in_row = True
            if not explicitly_parsed_terms and individual_terms_from_split and any(t.strip() for t in individual_terms_from_split):
                 found_unmappable_term_in_row = True


        if found_unmappable_term_in_row and 'other' in mapping_dict_for_lambda:
            codes.add(mapping_dict_for_lambda['other'])
            
        return sorted(list(codes))

    df['aspect_codes_list'] = df['aspect'].apply(
        lambda x: process_aspect_string(x, aspect_map_dict)
    )
    
    rows_before_filter = len(df)
    df = df[df['aspect_codes_list'].apply(lambda x: x != [aspect_map_dict['other']])] 

    rows_after_filter = len(df)
    print(f"Đã loại bỏ {rows_before_filter - rows_after_filter} dòng chỉ có aspect là '[{aspect_map_dict['other']}]' (other).")

    if df.empty:
        print("DataFrame trở nên rỗng sau khi lọc aspect '[0]'.")
        return None, None, None, None, None
        
    df.reset_index(drop=True, inplace=True) # Reset index sau khi lọc

    # 3. Tải Tokenizer
    print(f"Đang tải PhoBERT tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("Tokenizer đã được tải.")

    # 4. Tokenize cột 'reviews' (từ DataFrame đã được lọc)
    print("Đang tokenize cột 'reviews'...")
    review_texts = df['reviews'].tolist() 

    encoded_inputs = tokenizer(
        review_texts,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_inputs['input_ids']
    attention_masks = encoded_inputs['attention_mask']
    print(f"Đã tokenize reviews. Shape input_ids: {input_ids.shape}")

    # 5. Chuẩn bị nhãn 'sentiment' (từ DataFrame đã được lọc)
    print("Đang xử lý nhãn 'sentiment'...")
    adjusted_sentiment_labels = df['sentiment'].astype(int) + sentiment_adj
    sentiment_labels_tensor = torch.tensor(adjusted_sentiment_labels.tolist(), dtype=torch.long)
    print(f"Đã xử lý nhãn sentiment. Shape: {sentiment_labels_tensor.shape}")

    # 6. Chuẩn bị nhãn 'aspect' (từ DataFrame đã được lọc, cột aspect_codes_list đã có sẵn)
    print("Đang xử lý nhãn 'aspect' (multi-hot)...")
    aspect_labels_multi_hot_list = [
        create_multi_hot_vector(codes_list, num_aspects)
        for codes_list in df['aspect_codes_list'] # Sử dụng cột đã được lọc
    ]
    aspect_labels_tensor = torch.stack(aspect_labels_multi_hot_list)
    print(f"Đã xử lý nhãn aspect. Shape: {aspect_labels_tensor.shape}")

    # Tạo DataFrame metadata để lưu
    df_metadata = df[['reviews', 'sentiment', 'aspect', 'aspect_codes_list', 'type_product']].copy()

    return input_ids, attention_masks, sentiment_labels_tensor, aspect_labels_tensor, df_metadata

def save_prepared_data(input_ids, attention_masks, sentiment_labels, aspect_labels,
                       df_metadata, output_dir):
    if input_ids is None:
        print("Không có dữ liệu đã chuẩn bị để lưu.")
        return
    try:
        torch.save(input_ids, os.path.join(output_dir, "input_ids.pt"))
        torch.save(attention_masks, os.path.join(output_dir, "attention_masks.pt"))
        torch.save(sentiment_labels, os.path.join(output_dir, "sentiment_labels.pt"))
        torch.save(aspect_labels, os.path.join(output_dir, "aspect_labels.pt"))
        print(f"Đã lưu các tensor vào thư mục: {output_dir}")

        df_metadata.to_csv(os.path.join(output_dir, "finetuning_metadata.csv"), index=False)
        print(f"Đã lưu metadata vào: {os.path.join(output_dir, 'finetuning_metadata.csv')}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu đã chuẩn bị: {e}")

# --- Chạy chương trình ---
if __name__ == "__main__":
    print("=== BẮT ĐẦU QUÁ TRÌNH CHUẨN BỊ DỮ LIỆU CHO FINE-TUNING ===")

    input_ids_tensor, attention_masks_tensor, \
    sentiment_labels_tensor, aspect_labels_tensor, \
    metadata_df_final = prepare_data_for_finetuning(
        df_path=NORMALIZED_REVIEWS_FILE,
        tokenizer_name=PHOBERT_MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        aspect_map_dict=ASPECT_MAPPING, # Truyền dict mapping vào
        num_aspects=NUM_TOTAL_ASPECTS,
        sentiment_adj=SENTIMENT_ADJUSTMENT
    )

    if input_ids_tensor is not None:
        save_prepared_data(
            input_ids_tensor,
            attention_masks_tensor,
            sentiment_labels_tensor,
            aspect_labels_tensor,
            metadata_df_final,
            GOLD_DIR
        )
        print("\n--- Thông tin dữ liệu sau khi xử lý và lọc ---")
        print(f"Số lượng mẫu cuối cùng: {len(input_ids_tensor)}")
        if not metadata_df_final.empty:
            print("\n5 dòng đầu của metadata:")
            print(metadata_df_final.head())
            print("\nPhân phối sentiment cuối cùng:")
            print(metadata_df_final['sentiment'].value_counts(normalize=True).sort_index())
            
            print("\nPhân phối các aspect_codes_list (ví dụ 5 loại phổ biến nhất):")
            print(metadata_df_final['aspect_codes_list'].apply(lambda x: tuple(x) if isinstance(x, list) else x).value_counts().nlargest(5))

        print("Đã hoàn thành quá trình chuẩn bị dữ liệu cho fine-tuning.")
    else:
        print("Không thể chuẩn bị dữ liệu cho fine-tuning do lỗi hoặc không còn dữ liệu sau lọc.")
    
    print("=== KẾT THÚC QUÁ TRÌNH CHUẨN BỊ DỮ LIỆU ===")