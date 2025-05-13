import os
import sys
import json
import traceback
import random
import re
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import torch
from bs4 import BeautifulSoup
import requests
import time

# --- Thêm các import cho Selenium ---
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager

# Thêm đường dẫn cơ sở của dự án vào sys.path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

# Import các module từ dự án
from src.utils.constants import GOLD_DIR, MODELS_DIR, BRONZE_DIR, SILVER_DIR
from src.data_preprocessing.normalize_text import normalize_text
from src.data_preprocessing.clean_data import clean_text
from src.data_preprocessing.tokenize_text import tokenize_text
from src.data_embedding.sentiment_aspect_extraction import SentimentAspectPredictor

BROWSER_PROFILES_DIR = os.path.join(current_dir, "browser_profiles")
if not os.path.exists(BROWSER_PROFILES_DIR):
    os.makedirs(BROWSER_PROFILES_DIR, exist_ok=True)

def setup_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--disable-gpu") 
    chrome_options.add_argument("--no-sandbox") 
    chrome_options.add_argument("--disable-dev-shm-usage")
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        return driver
    except Exception as e:
        print(f"Lỗi khi thiết lập Selenium driver: {e}")
        traceback.print_exc()
        return None

def click_show_more_buttons_selenium(driver):
    try:
        WebDriverWait(driver, 2).until(
            EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@class, 'review-comment__content--less')]//span[contains(text(), 'Xem thêm')] | //span[@class='show-more-content' and text()='Xem thêm']"))
        )
        show_more_buttons = driver.find_elements(By.XPATH, "//div[contains(@class, 'review-comment__content--less')]//span[contains(text(), 'Xem thêm')] | //span[@class='show-more-content' and text()='Xem thêm']")
        for button in show_more_buttons:
            try:
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", button)
                time.sleep(0.2) 
                driver.execute_script("arguments[0].click();", button)
                time.sleep(0.3) 
            except StaleElementReferenceException:
                print("Nút 'Xem thêm' đã cũ, thử tìm lại.") 
            except Exception as e_click:
                print(f"Lỗi khi nhấn nút 'Xem thêm': {e_click}") 
    except TimeoutException:
        print("Không tìm thấy nút 'Xem thêm' nào hoặc đã hết thời gian chờ.")
    except Exception as e:
        print(f"Lỗi khi xử lý nút 'Xem thêm': {e}")

def extract_review_data_selenium(driver):
    reviews = []
    try:
        review_containers = driver.find_elements(By.CSS_SELECTOR, "div.review-comment")
        for container in review_containers:
            title = "N/A"
            content = "N/A"
            try:
                title_element = container.find_element(By.CSS_SELECTOR, "div.review-comment__title")
                title = title_element.text.strip()
            except NoSuchElementException:
                pass
            try:
                content_element_full = container.find_element(By.CSS_SELECTOR, "div.review-comment__content.review-comment__content--expanded span")
                content = content_element_full.text.strip()
            except NoSuchElementException:
                try:
                    content_element_normal = container.find_element(By.CSS_SELECTOR, "div.review-comment__content")
                    content = content_element_normal.text.strip()
                    if "Xem thêm" in content and content.endswith("Xem thêm"):
                        parts = content.rsplit("Xem thêm", 1)
                        if parts[0].strip(): 
                            content = parts[0].strip()
                        else: 
                            try:
                                parent_content_element = container.find_element(By.XPATH, "./div[contains(@class,'review-comment__content-wrapper')]/div[contains(@class,'review-comment__content')]")
                                content = parent_content_element.text.strip()
                                if "Xem thêm" in content and content.endswith("Xem thêm"):
                                     content = content.rsplit("Xem thêm", 1)[0].strip()
                            except NoSuchElementException:
                                pass 
                except NoSuchElementException:
                    pass             
            if title == "N/A" and content == "N/A":
                continue 
            reviews.append({"title": title, "content": content})
    except Exception as e:
        print(f"Lỗi khi trích xuất dữ liệu đánh giá bằng Selenium: {e}")
        traceback.print_exc()
    return reviews

def navigate_and_extract_reviews_selenium(driver, url):
    all_reviews = []
    try:
        print(f"Truy cập URL: {url}")
        driver.get(url)
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
        time.sleep(2) 
        try:
            reviews_section_wrapper = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'customer-reviews') or @id='customer-reviews' or //h2[contains(text(),'Đánh giá') or contains(text(),'Nhận xét') or contains(text(),'Bình luận')]]"))
            )
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", reviews_section_wrapper)
            print("Đã cuộn đến khu vực đánh giá.")
            time.sleep(2)
        except TimeoutException:
            print("Không tìm thấy khu vực đánh giá cụ thể, sẽ cuộn từ từ xuống.")
            for _ in range(3): 
                driver.execute_script("window.scrollBy(0, window.innerHeight * 0.8);")
                time.sleep(1.5)
        
        page_num = 1
        max_pages = 10 
        
        while page_num <= max_pages:
            print(f"Đang xử lý trang đánh giá số: {page_num}")
            driver.execute_script("window.scrollBy(0, 300);")
            time.sleep(1)
            click_show_more_buttons_selenium(driver) 
            page_reviews = extract_review_data_selenium(driver)
            if page_reviews:
                all_reviews.extend(page_reviews)
                print(f"Trích xuất được {len(page_reviews)} đánh giá từ trang {page_num}. Tổng: {len(all_reviews)}")
            else:
                print(f"Không có đánh giá nào trên trang {page_num}.")
            try:
                next_button_xpath_options = [
                    "//ul[contains(@class,'review-pages')]//a[contains(@class,'next') and not(contains(@class,'disabled'))]",
                    "//a[@class='btn next' and not(contains(@class,'disabled'))]", 
                    "//button[(@class='btn-next' or contains(text(),'Sau') or contains(text(),'Next') or contains(@class,'next')) and not(@disabled)]",
                    "//li[contains(@class,'next') and not(contains(@class,'disabled'))]/a"
                ]
                next_button = None
                for xpath in next_button_xpath_options:
                    try:
                        candidate_button = WebDriverWait(driver, 7).until(EC.element_to_be_clickable((By.XPATH, xpath)))
                        if "disabled" in candidate_button.get_attribute("class").lower() or candidate_button.get_attribute("disabled"):
                            continue
                        next_button = candidate_button
                        print(f"Tìm thấy nút 'Next' với xpath: {xpath}")
                        break
                    except (TimeoutException, NoSuchElementException):
                        pass
                if not next_button:
                    print("Không tìm thấy nút 'Next' hoặc đã đến trang cuối cùng.")
                    break
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", next_button)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(3) 
                page_num += 1
            except StaleElementReferenceException:
                print("Nút 'Next' đã cũ (stale).")
                break 
            except TimeoutException:
                print("Không tìm thấy nút 'Next' hoặc đã đến trang cuối cùng (Timeout).")
                break
            except Exception as e_nav:
                print(f"Lỗi khi điều hướng sang trang tiếp theo: {e_nav}")
                traceback.print_exc()
                break
        if page_num > max_pages:
            print(f"Đã đạt giới hạn tối đa {max_pages} trang đánh giá.")
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi xử lý URL {url} bằng Selenium: {e}")
        traceback.print_exc()
    print(f"Hoàn tất thu thập cho URL. Tổng số đánh giá: {len(all_reviews)}")
    return all_reviews

def crawl_product_reviews(url):
    print(f"Bắt đầu thu thập đánh giá từ: {url} bằng Selenium")
    reviews = []
    driver = None
    try:
        driver = setup_selenium_driver()
        if driver:
            reviews = navigate_and_extract_reviews_selenium(driver, url)
        else:
            print("Không thể khởi tạo Selenium driver. Bỏ qua thu thập đánh giá.")
    except Exception as e:
        print(f"Lỗi trong quá trình thu thập đánh giá bằng Selenium: {e}")
        traceback.print_exc()
    finally:
        if driver:
            print("Đang đóng Selenium driver...")
            driver.quit()
            print("Đã đóng Selenium driver.")
    if not reviews:
        reviews = [{
            "title": "Không tìm thấy đánh giá",
            "content": "Website có thể đã thay đổi cấu trúc, không có đánh giá nào cho sản phẩm này, hoặc có lỗi khi dùng Selenium.",
            "product_url": url 
        }]
    return reviews

# --- CẬP NHẬT HÀM extract_product_info DỰA TRÊN HTML MỚI ---
def extract_product_info(url):
    """
    Trích xuất thông tin sản phẩm từ URL Tiki (tên, giá, rating, số review, ảnh, v.v.).
    Sử dụng requests và BeautifulSoup.
    """
    print(f"Bắt đầu trích xuất thông tin sản phẩm từ: {url} bằng Requests/BeautifulSoup")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36', # User agent cập nhật
            'Accept-Language': 'vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # --- Tên sản phẩm ---
        product_name = "Không có thông tin"
        # Ưu tiên selector mới từ HTML bạn cung cấp
        name_selectors = [
            "h1.sc-c0f8c612-0", # Selector từ HTML mới (có thể bỏ phần hash dEurho)
            "h1[class*='sc-c0f8c612-0']", # Regex cho class nếu phần hash thay đổi
            # Các selectors cũ hơn làm fallback
            "h1.title", "h1.heading", "h1.pdp-name", "div.header h1", 
            "h1[data-view-id='pdp_product_name']", "div.product-name h1",
            "span.product-name"
        ]
        for selector in name_selectors:
            name_elem = soup.select_one(selector)
            if name_elem:
                product_name = name_elem.text.strip()
                print(f"Tìm thấy tên sản phẩm với selector: {selector}")
                break
        
        # --- Giá hiện tại ---
        price = "Không có thông tin"
        # Selector từ HTML mới: div.product-price__current-price (đã có)
        price_selectors = [
            "div.product-price__current-price", 
            "div.price-and-icon span", "div.flash-sale-price span", 
            "div.style__Price-sc-1aml21n-6", "div.product-price span", 
            "span.current-price", "div[data-view-id='pdp_price'] .price", 
            "span.price"
        ]
        for selector in price_selectors:
            price_elem = soup.select_one(selector)
            if price_elem:
                price = price_elem.text.strip().replace('\xa0₫', '₫').replace('.', '').replace('₫', '').strip() + '₫'
                print(f"Tìm thấy giá sản phẩm với selector: {selector} -> {price}")
                break

        # --- Giá gốc ---
        original_price = "N/A"
        original_price_elem = soup.select_one("div.product-price__current-price del")
        if not original_price_elem: # Thử selector khác nếu cần
            original_price_elem = soup.select_one("div.product-price__original-price")

        if original_price_elem:
            original_price_text = original_price_elem.text.strip().replace('\xa0₫', '₫').replace('.', '').replace('₫', '').strip()
            if original_price_text:
                 original_price = original_price_text + '₫'
            print(f"Tìm thấy giá gốc: {original_price}")
        
        # --- Tỷ lệ giảm giá ---
        discount_rate = "N/A"
        discount_rate_elem = soup.select_one("div.product-price__discount-rate")
        if discount_rate_elem:
            discount_rate = discount_rate_elem.text.strip()
            print(f"Tìm thấy tỷ lệ giảm giá: {discount_rate}")

        # --- Rating ---
        rating = 0.0
        # Selector từ HTML mới: div.sc-1a46a934-1.dCjKzJ > div (lấy text của div con đầu tiên)
        rating_selectors = [
            "div.sc-1a46a934-1 > div:first-child", # Selector từ HTML mới
            "div.review-rating__point", "div.product-rating span", 
            "div.rating-star span", "span.rating-point",
            "div[data-view-id='pdp_rating_summary'] span.value"
        ]
        for selector in rating_selectors:
            rating_elem = soup.select_one(selector)
            if rating_elem:
                try:
                    rating_text = rating_elem.text.strip()
                    match = re.search(r'(\d+(\.\d+)?)', rating_text) # Lấy số float đầu tiên
                    if match:
                        rating = float(match.group(1))
                        print(f"Tìm thấy rating với selector: {selector}, giá trị: {rating}")
                        break
                except Exception:
                    pass
        
        # --- Số lượng đánh giá ---
        review_count = 0
        # Selector từ HTML mới: a.number[data-view-id="pdp_main_view_review"]
        review_count_selectors = [
            "a.number[data-view-id='pdp_main_view_review']", # Selector từ HTML mới
            "div.review-rating__total", "div.product-rating-summary", 
            "div.rating-count", "a[data-view-id='pdp_rating_summary_review_count_link']"
        ]
        for selector in review_count_selectors:
            review_count_elem = soup.select_one(selector)
            if review_count_elem:
                count_text = review_count_elem.text.strip()
                match = re.search(r'(\d+)', count_text) 
                if match:
                    review_count = int(match.group(1))
                    print(f"Tìm thấy số lượng đánh giá với selector: {selector}, giá trị: {review_count}")
                    break
        
        # --- Số lượng đã bán ---
        quantity_sold = "N/A"
        quantity_sold_elem = soup.select_one("div[data-view-id='pdp_quantity_sold']")
        if quantity_sold_elem:
            sold_text = quantity_sold_elem.text.strip() # Ví dụ: "Đã bán 134"
            match = re.search(r'(\d+)', sold_text)
            if match:
                quantity_sold = int(match.group(1))
            else: # Nếu chỉ có text "Đã bán xxx" mà không có số cụ thể
                quantity_sold = sold_text # Giữ lại text gốc
            print(f"Tìm thấy số lượng đã bán: {quantity_sold}")

        # --- Hình ảnh ---
        image_url = ""
        # Selector từ HTML mới: picture.webpimg-container img (đã có, nhưng ưu tiên srcset)
        # Hoặc picture.webpimg-container source[type="image/webp"]
        image_selectors = [
            "div.image-frame picture.webpimg-container source[type='image/webp']", # Selector mới từ HTML của người dùng
            "div.image-frame div[data-view-id='pdp_main_view_gallery'] picture.webpimg-container source[type='image/webp']", # Selector chi tiết hơn
            "picture.webpimg-container source[type='image/webp']", # Ưu tiên source webp
            "picture.webpimg-container img",
            "div.thumbnail img", "div.style__StyledSlideshow-sc-x7f02x-0 img",
            "div.product-image img", "div[data-view-id='pdp_main_image'] img",
            "img.styles__StyledImg-sc-p9s3t3-0"
        ]
        for selector in image_selectors:
            img_elem = soup.select_one(selector)
            if img_elem:
                srcset = img_elem.get('srcset')
                src = img_elem.get('src')
                if srcset:
                    # Lấy URL đầu tiên từ srcset (thường là đủ tốt)
                    image_url = srcset.split(',')[0].split(' ')[0] 
                elif src:
                    image_url = src
                
                if image_url:
                    if not image_url.startswith('http'):
                        base_url_match = re.match(r'^(https?://[^/]+)', url)
                        if base_url_match:
                            image_url = base_url_match.group(1) + image_url
                    print(f"Tìm thấy hình ảnh với selector: {selector} -> {image_url}")
                    break
        
        result = {
            'name': product_name, 
            'price': price,
            'original_price': original_price,
            'discount_rate': discount_rate,
            'rating': rating,
            'reviewCount': review_count,
            'quantity_sold': quantity_sold,
            'image': image_url
        }
        print(f"Đã trích xuất thông tin sản phẩm: {json.dumps(result, ensure_ascii=False)}")
        return result
    
    except requests.exceptions.RequestException as e_req:
        print(f"Lỗi Request khi trích xuất thông tin sản phẩm ({url}): {e_req}")
    except Exception as e:
        print(f"Lỗi không xác định khi trích xuất thông tin sản phẩm ({url}): {e}")
        traceback.print_exc()
        
    return { 
        'name': 'Không thể lấy thông tin', 'price': 'Không có thông tin',
        'original_price': 'N/A', 'discount_rate': 'N/A',
        'rating': 0, 'reviewCount': 0, 'quantity_sold': 'N/A',
        'image': ''
    }

# --- Các phần còn lại của Flask App (MODEL_PATH, MAPPINGS, routes, v.v.) ---
# --- Giữ nguyên như phiên bản trước đã tích hợp Selenium ---

app = Flask(__name__, static_folder='static', template_folder='templates')

MODEL_PATH = os.path.join(MODELS_DIR, "phobert_finetuned", "best_model.pt")
SENTIMENT_MAPPING = {0: 'rất tiêu cực', 1: 'tiêu cực', 2: 'trung lập', 3: 'tích cực', 4: 'rất tích cực'}
ASPECT_MAPPING = {0: 'other', 1: 'cskh', 2: 'quality', 3: 'price', 4: 'ship'}

if not os.path.exists(MODEL_PATH):
    print(f"CẢNH BÁO: Không tìm thấy mô hình tại {MODEL_PATH}")
    print("Vui lòng chạy quá trình fine-tune trước khi khởi động ứng dụng.")
    predictor = None
else:
    try:
        predictor = SentimentAspectPredictor(MODEL_PATH)
        print("Đã tải mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        predictor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url')
    
    if not url or not (url.startswith('https://tiki.vn') or url.startswith('http://tiki.vn')):
        return jsonify({'error': 'URL không hợp lệ. Chỉ hỗ trợ URL từ tiki.vn'}), 400
    
    if predictor is None:
         return jsonify({'error': 'Mô hình phân tích chưa được tải. Vui lòng kiểm tra logs.'}), 500

    try:
        product_info = extract_product_info(url) # Đã cập nhật hàm này
        reviews_raw = crawl_product_reviews(url) 
        
        if len(reviews_raw) == 1 and reviews_raw[0]["title"] == "Không tìm thấy đánh giá":
            print("Không có đánh giá thực tế để phân tích.")
            return jsonify({
                'product': product_info,
                'sentiment': default_sentiment(),
                'aspects': default_aspects(),
                'recommendation': {
                    'shouldBuy': False,
                    'text': reviews_raw[0]["content"] 
                },
                'exampleReviews': [],
                'raw_reviews_count': 0,
                'analyzed_reviews_count': 0
            })

        reviews_df = pd.DataFrame(reviews_raw)
        reviews_df = reviews_df[reviews_df['content'].notna() & (reviews_df['content'].str.strip() != '') & (reviews_df['content'] != 'N/A')]

        if reviews_df.empty:
            print("DataFrame đánh giá rỗng sau khi lọc.")
            return jsonify({
                'product': product_info,
                'sentiment': default_sentiment(),
                'aspects': default_aspects(),
                'recommendation': {'shouldBuy': False, 'text': 'Không có nội dung đánh giá hợp lệ để phân tích.'},
                'exampleReviews': [],
                'raw_reviews_count': len(reviews_raw),
                'analyzed_reviews_count': 0
            })

        print(f"Số lượng đánh giá thô (có nội dung): {len(reviews_df)}")

        reviews_df['content_cleaned'] = reviews_df['content'].apply(clean_text)
        reviews_df['normalized_text'] = reviews_df['content_cleaned'].apply(normalize_text)
        reviews_df['tokenized_text'] = reviews_df['normalized_text'].apply(tokenize_text)
        
        reviews_df = reviews_df[reviews_df['tokenized_text'].str.strip() != '']
        if reviews_df.empty:
             print("DataFrame đánh giá rỗng sau khi tokenize và lọc.")
             return jsonify({
                'product': product_info,
                'sentiment': default_sentiment(),
                'aspects': default_aspects(),
                'recommendation': {'shouldBuy': False, 'text': 'Nội dung đánh giá không thể xử lý sau khi tokenize.'},
                'exampleReviews': [],
                'raw_reviews_count': len(reviews_raw),
                'analyzed_reviews_count': 0
            })

        tokenized_texts = reviews_df['tokenized_text'].tolist()
        predictions = predictor.predict_and_format(tokenized_texts)
        
        reviews_df['sentiment'] = [pred['sentiment_label'] for pred in predictions]
        reviews_df['sentiment_text_mapped'] = [pred['sentiment'] for pred in predictions]
        
        for aspect in ASPECT_MAPPING.values():
            reviews_df[f'aspect_{aspect}'] = 0
        
        for i, pred in enumerate(predictions):
            df_index = reviews_df.index[i] 
            for aspect_name in pred['aspects']:
                reviews_df.loc[df_index, f'aspect_{aspect_name}'] = 1
        
        sentiment_counts = reviews_df['sentiment'].value_counts().to_dict()
        total_reviews_analyzed = len(reviews_df)
        
        sentiment_percentages = {
            'veryPositive': round(sentiment_counts.get(4, 0) / total_reviews_analyzed * 100) if total_reviews_analyzed > 0 else 0,
            'positive': round(sentiment_counts.get(3, 0) / total_reviews_analyzed * 100) if total_reviews_analyzed > 0 else 0,
            'neutral': round(sentiment_counts.get(2, 0) / total_reviews_analyzed * 100) if total_reviews_analyzed > 0 else 0,
            'negative': round(sentiment_counts.get(1, 0) / total_reviews_analyzed * 100) if total_reviews_analyzed > 0 else 0,
            'veryNegative': round(sentiment_counts.get(0, 0) / total_reviews_analyzed * 100) if total_reviews_analyzed > 0 else 0
        }
        
        aspect_scores = {}
        for aspect_label, aspect_name_key in ASPECT_MAPPING.items():
            if aspect_name_key == 'other': continue
            aspect_reviews = reviews_df[reviews_df[f'aspect_{aspect_name_key}'] == 1]
            if len(aspect_reviews) > 0:
                positive_count = len(aspect_reviews[aspect_reviews['sentiment'] >= 3]) 
                aspect_scores[aspect_name_key] = round(positive_count / len(aspect_reviews) * 100)
            else:
                aspect_scores[aspect_name_key] = 0 
        
        positive_percentage = sentiment_percentages['veryPositive'] + sentiment_percentages['positive']
        negative_percentage = sentiment_percentages['veryNegative'] + sentiment_percentages['negative']
        
        should_buy = positive_percentage >= 65 and negative_percentage <= 25 
        recommendation_text = "Cân nhắc kỹ trước khi mua. Sản phẩm có nhiều ý kiến trái chiều."
        if should_buy:
            recommendation_text = "Nên mua! Đa số người dùng hài lòng với sản phẩm này."
        if negative_percentage >= 40: 
            recommendation_text = "Không nên mua! Nhiều người dùng không hài lòng với sản phẩm."
            should_buy = False
        if total_reviews_analyzed == 0:
             recommendation_text = "Không có đánh giá nào được phân tích để đưa ra khuyến nghị."
             should_buy = False # Hoặc một trạng thái trung lập
        elif total_reviews_analyzed < 5 : 
            recommendation_text = "Sản phẩm có quá ít đánh giá để đưa ra nhận xét đáng tin cậy. Cần thêm thông tin."
        
        example_reviews_output = []
        reviews_df_sorted_pos = reviews_df.sort_values(by='sentiment', ascending=False)
        reviews_df_sorted_neg = reviews_df.sort_values(by='sentiment', ascending=True)

        if not reviews_df_sorted_pos.empty:
            top_positive = reviews_df_sorted_pos.iloc[0]
            example_reviews_output.append({
                'text': top_positive['content'], 
                'sentiment': int(top_positive['sentiment']),
                'sentimentText': SENTIMENT_MAPPING[int(top_positive['sentiment'])],
                'aspects': [asp.replace('aspect_', '') for asp in top_positive.index 
                            if asp.startswith('aspect_') and top_positive[asp] == 1]
            })
        
        if not reviews_df_sorted_neg.empty:
            top_negative = reviews_df_sorted_neg.iloc[0]
            if len(example_reviews_output) == 0 or (len(example_reviews_output) > 0 and example_reviews_output[0]['text'] != top_negative['content']):
                example_reviews_output.append({
                    'text': top_negative['content'],
                    'sentiment': int(top_negative['sentiment']),
                    'sentimentText': SENTIMENT_MAPPING[int(top_negative['sentiment'])],
                    'aspects': [asp.replace('aspect_', '') for asp in top_negative.index 
                                if asp.startswith('aspect_') and top_negative[asp] == 1]
                })
        
        if len(reviews_df) > len(example_reviews_output):
            remaining_reviews = reviews_df[~reviews_df['content'].isin([ex['text'] for ex in example_reviews_output])]
            if not remaining_reviews.empty:
                random_review = remaining_reviews.sample(1).iloc[0]
                example_reviews_output.append({
                    'text': random_review['content'],
                    'sentiment': int(random_review['sentiment']),
                    'sentimentText': SENTIMENT_MAPPING[int(random_review['sentiment'])],
                    'aspects': [asp.replace('aspect_', '') for asp in random_review.index 
                                if asp.startswith('aspect_') and random_review[asp] == 1]
                })
        
        return jsonify({
            'product': product_info,
            'sentiment': sentiment_percentages,
            'aspects': aspect_scores,
            'recommendation': {'shouldBuy': should_buy, 'text': recommendation_text},
            'exampleReviews': example_reviews_output,
            'raw_reviews_count': len(reviews_raw), 
            'analyzed_reviews_count': total_reviews_analyzed
        })
    
    except Exception as e:
        print(f"Lỗi tổng thể trong API /analyze: {e}")
        traceback.print_exc()
        default_product_info = {'name': 'Lỗi xử lý', 'price': '', 'rating': 0, 'reviewCount': 0, 'image': ''}
        if 'product_info' in locals() and product_info: 
             default_product_info = product_info
        return jsonify({
            'error': f'Lỗi máy chủ nội bộ: {str(e)}',
            'product': default_product_info,
            'sentiment': default_sentiment(),
            'aspects': default_aspects(),
            'recommendation': {'shouldBuy': False, 'text': 'Đã xảy ra lỗi trong quá trình phân tích.'},
            'exampleReviews': []
            }), 500

def default_sentiment():
    return {'veryPositive': 0, 'positive': 0, 'neutral': 0, 'negative': 0, 'veryNegative': 0}

def default_aspects():
    return {aspect_name: 0 for aspect_label, aspect_name in ASPECT_MAPPING.items() if aspect_name != 'other'}

if __name__ == '__main__':
    app.run(debug=True, port=5000)