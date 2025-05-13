import pandas as pd
import time
import sys
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.constants import PRODUCT_URLS_FILE, RAW_REVIEWS_FILE, BRONZE_DIR
from src.utils.driver_setup import setup_driver

def click_show_more_buttons(driver):
    """
    Click các nút "Xem thêm" để hiển thị toàn bộ nội dung đánh giá.
    
    Args:
        driver: Selenium WebDriver
    """
    try:
        show_more_buttons = driver.find_elements(By.XPATH, "//span[@class='show-more-content' and text()='Xem thêm']")
        for button in show_more_buttons:
            try:
                driver.execute_script("arguments[0].click();", button)
                time.sleep(0.5)
            except Exception:
                pass
    except Exception as e:
        print(f"Lỗi khi click nút 'Xem thêm': {e}")

def extract_review_data(driver):
    """
    Trích xuất dữ liệu đánh giá từ trang hiện tại.
    
    Args:
        driver: Selenium WebDriver
        
    Returns:
        list: Danh sách các đánh giá trích xuất được
    """
    reviews = []
    try:
        review_containers = driver.find_elements(By.CSS_SELECTOR, "div.review-comment")
        
        for container in review_containers:
            try:
                # Trích xuất tiêu đề
                title = "N/A"
                try:
                    title_element = container.find_element(By.CSS_SELECTOR, "div.review-comment__title")
                    title = title_element.text.strip()
                except NoSuchElementException:
                    pass
                
                # Trích xuất nội dung
                content = "N/A"
                try:
                    content_element = container.find_element(By.CSS_SELECTOR, "div.review-comment__content")
                    content = content_element.text.strip()
                except NoSuchElementException:
                    pass
                
                # Thêm đánh giá vào danh sách
                # Chỉ thêm nếu có nội dung hợp lệ
                if title != "N/A" or content != "N/A":
                    reviews.append({"title": title, "content": content})
                
            except Exception as e:
                print(f"Lỗi khi trích xuất dữ liệu đánh giá: {e}")
                continue
                
    except Exception as e:
        print(f"Lỗi khi tìm container đánh giá: {e}")
    
    return reviews

def navigate_through_reviews(driver, url):
    """
    Điều hướng qua các trang đánh giá và thu thập dữ liệu.
    
    Args:
        driver: Selenium WebDriver
        url (str): URL sản phẩm
        
    Returns:
        list: Danh sách tất cả đánh giá thu thập được
    """
    all_reviews = []
    
    try:
        driver.get(url)
        time.sleep(3)  # Chờ trang tải
        
        # Cuộn đến phần đánh giá
        try:
            reviews_section = driver.find_element(By.ID, "productReviews")
            driver.execute_script("arguments[0].scrollIntoView();", reviews_section)
            time.sleep(2)
        except NoSuchElementException:
            print("Không tìm thấy phần đánh giá, cuộn xuống dưới")
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(2)
        
        page_num = 1
        while True:
            print(f"Đang xử lý trang đánh giá {page_num}")
            
            # Click các nút "Xem thêm" để hiển thị toàn bộ nội dung
            click_show_more_buttons(driver)
            
            # Trích xuất đánh giá từ trang hiện tại
            page_reviews = extract_review_data(driver)
            if page_reviews:
                all_reviews.extend(page_reviews)
                print(f"Đã trích xuất {len(page_reviews)} đánh giá từ trang {page_num}")
            
            # Tìm và click nút chuyển trang tiếp theo
            try:
                next_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn.next"))
                )
                
                # Kiểm tra xem nút tiếp theo có bị vô hiệu hóa không
                if next_button.get_attribute("class") and "disabled" in next_button.get_attribute("class"):
                    print("Đã đến trang cuối cùng")
                    break
                
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(2)  # Chờ trang tiếp theo tải
                page_num += 1
            except (TimeoutException, StaleElementReferenceException):
                print("Không tìm thấy nút tiếp theo hoặc đã đến trang cuối")
                break
            except Exception as e:
                print(f"Lỗi khi điều hướng đến trang tiếp theo: {e}")
                break
    
    except Exception as e:
        print(f"Lỗi khi xử lý URL {url}: {e}")
    
    return all_reviews

def load_product_urls():
    """
    Đọc file CSV chứa URL sản phẩm.
    
    Returns:
        list: Danh sách URL sản phẩm.
    """
    try:
        df = pd.read_csv(PRODUCT_URLS_FILE)
        url_column = [col for col in df.columns if 'url' in col.lower()][0]
        product_urls = df[url_column].tolist()
        print(f"Đã đọc {len(product_urls)} URL sản phẩm từ {PRODUCT_URLS_FILE}")
        return product_urls
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {PRODUCT_URLS_FILE}")
        return []
    except Exception as e:
        print(f"Lỗi khi đọc file {PRODUCT_URLS_FILE}: {e}")
        return []

def crawl_product_reviews():
    """
    Thu thập đánh giá từ tất cả URL sản phẩm.
    
    Returns:
        list: Danh sách tất cả đánh giá thu thập được
    """
    product_urls = load_product_urls()
    if not product_urls:
        return []
    
    all_reviews = []
    driver = setup_driver()
    
    try:
        for index, url in enumerate(product_urls):
            print(f"\n--- Đang xử lý URL sản phẩm {index + 1}/{len(product_urls)} ---")
            
            reviews = navigate_through_reviews(driver, url)
            if reviews:
                # Thêm URL sản phẩm vào mỗi đánh giá
                for review in reviews:
                    review['product_url'] = url
                
                all_reviews.extend(reviews)
                
                # Lưu kết quả trung gian sau mỗi sản phẩm
                temp_df = pd.DataFrame(all_reviews)
                temp_output = os.path.join(BRONZE_DIR, f"reviews_temp_{index+1}.csv")
                temp_df.to_csv(temp_output, index=False)
                
            print(f"Đã xử lý URL {index+1}/{len(product_urls)}. Tổng số đánh giá hiện tại: {len(all_reviews)}")
                
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        driver.quit()
    
    return all_reviews

def save_reviews(reviews):
    """
    Lưu đánh giá vào file CSV.
    
    Args:
        reviews (list): Danh sách các đánh giá cần lưu.
    """
    if not reviews:
        print("Không có đánh giá nào để lưu.")
        return
    
    try:
        df = pd.DataFrame(reviews)
        df.to_csv(RAW_REVIEWS_FILE, index=False)
        print(f"Đã lưu {len(reviews)} đánh giá vào {RAW_REVIEWS_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu đánh giá: {e}")

def main():
    """Hàm chính để thu thập và lưu đánh giá sản phẩm."""
    reviews = crawl_product_reviews()
    save_reviews(reviews)

if __name__ == "__main__":
    main() 