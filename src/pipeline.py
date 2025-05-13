import argparse
import os
import sys
import time

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection.crawl_category_urls import main as crawl_category_urls
from src.data_collection.crawl_product_urls import main as crawl_product_urls
from src.data_collection.crawl_product_reviews import main as crawl_product_reviews
from src.data_preprocessing.clean_data import main as clean_data
from src.data_labeling.label_aspects import main as label_aspects
from src.data_preprocessing.normalize_text import main as normalize_text
from src.data_preprocessing.normalize_text import main as normalize_data
from src.data_embedding.phobert_embedding import main as create_embeddings

def run_pipeline(steps=None):
    """
    Chạy pipeline xử lý dữ liệu.
    
    Args:
        steps (list, optional): Danh sách các bước cần chạy. Nếu None, chạy tất cả các bước.
    """
    all_steps = {
        # Bước 1: Thu thập dữ liệu
        # 'crawl_category': crawl_category_urls,
        # 'crawl_products': crawl_product_urls,
        # 'crawl_reviews': crawl_product_reviews,
        
        # Bước 2: Tiền xử lý (làm sạch) dữ liệu
        'clean_data': clean_data,
        
        # Bước 3: Gắn nhãn cho dữ liệu
        'label_aspects': label_aspects,
        
        # Bước 4: Chuẩn hóa văn bản
        'normalize_text': normalize_text,
        'normalize_data': normalize_data,
        
        # Bước 5: Tạo embedding với PhoBERT
        'create_embeddings': create_embeddings
    }
    
    if steps is None:
        steps = list(all_steps.keys())
    
    start_time = time.time()
    
    for step in steps:
        if step not in all_steps:
            print(f"Bước '{step}' không tồn tại. Bỏ qua.")
            continue
        
        print(f"\n{'=' * 50}")
        print(f"Bắt đầu bước: {step}")
        print(f"{'=' * 50}\n")
        
        step_start_time = time.time()
        
        try:
            all_steps[step]()
        except Exception as e:
            print(f"Lỗi khi thực hiện bước '{step}': {e}")
        
        step_end_time = time.time()
        print(f"\nHoàn thành bước '{step}' trong {step_end_time - step_start_time:.2f} giây.")
    
    end_time = time.time()
    print(f"\n{'=' * 50}")
    print(f"Hoàn thành pipeline trong {end_time - start_time:.2f} giây.")
    print(f"{'=' * 50}")

def parse_args():
    """
    Phân tích tham số dòng lệnh.
    
    Returns:
        argparse.Namespace: Đối tượng chứa các tham số dòng lệnh.
    """
    parser = argparse.ArgumentParser(description='Pipeline xử lý dữ liệu đánh giá Tiki')
    
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=[
            'crawl_category',
            'crawl_products',
            'crawl_reviews',
            'clean_data',
            'label_aspects',
            'normalize_text',
            'create_embeddings'
        ],
        help='Các bước cần chạy (mặc định: chạy tất cả các bước)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.steps) 