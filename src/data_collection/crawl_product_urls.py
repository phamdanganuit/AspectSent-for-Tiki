import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException
from urllib.parse import urljoin
import time
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.constants import CATEGORY_URLS_FILE, PRODUCT_URLS_FILE
from src.utils.driver_setup import setup_driver

def load_category_urls():
    """
    Đọc file CSV chứa URL danh mục.
    
    Returns:
        list: Danh sách URL danh mục.
    """
    try:
        df = pd.read_csv(CATEGORY_URLS_FILE)
        category_urls = df['URL'].tolist()
        print(f"Đã đọc {len(category_urls)} URL danh mục từ {CATEGORY_URLS_FILE}")
        return category_urls
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {CATEGORY_URLS_FILE}")
        return []
    except Exception as e:
        print(f"Lỗi khi đọc file {CATEGORY_URLS_FILE}: {e}")
        return []

def crawl_products_from_category(driver, category_url, max_load_more_clicks=3):
    """
    Thu thập URL sản phẩm từ một URL danh mục cụ thể.
    
    Args:
        driver: Selenium WebDriver
        category_url (str): URL danh mục cần crawl
        max_load_more_clicks (int): Số lần tối đa nhấp nút "Xem thêm"
        
    Returns:
        set: Tập hợp URL sản phẩm thu thập được
    """
    print(f"Đang xử lý URL danh mục: {category_url}")
    base_url = "https://tiki.vn"
    product_links = set()
    xem_them_button_selector = "div[data-view-id='category_infinity_view.more']"
    product_link_selector = "a.product-item"
    
    driver.get(category_url)
    time.sleep(2)
    
    # Click "Xem thêm" để tải thêm sản phẩm
    actual_clicks = 0
    for i in range(max_load_more_clicks):
        try:
            wait = WebDriverWait(driver, 10)
            # Cuộn xuống cuối trang
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.8)
            
            xem_them_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, xem_them_button_selector)))
            driver.execute_script("arguments[0].click();", xem_them_button)
            actual_clicks += 1
            print(f"  -> Đã nhấp 'Xem thêm' lần thứ {actual_clicks}")
            time.sleep(3)
        except (TimeoutException, NoSuchElementException, ElementNotInteractableException):
            print("  -> Không tìm thấy hoặc không thể nhấp 'Xem thêm' nữa.")
            break
        except Exception as e:
            print(f"  -> Lỗi khi nhấp 'Xem thêm': {e}")
            break
    
    # Thu thập URL sản phẩm
    try:
        product_elements = driver.find_elements(By.CSS_SELECTOR, product_link_selector)
        print(f"  -> Tìm thấy {len(product_elements)} phần tử sản phẩm.")
        
        for element in product_elements:
            try:
                href = element.get_attribute('href')
                if href:
                    full_url = urljoin(base_url, href)
                    product_links.add(full_url)
            except Exception as e:
                print(f"  -> Lỗi khi lấy href: {e}")
        
        print(f"  -> Thu thập được {len(product_links)} URL sản phẩm duy nhất.")
    except Exception as e:
        print(f"  -> Lỗi khi thu thập URL sản phẩm: {e}")
    
    return product_links

def crawl_product_urls(max_load_more_clicks=3):
    """
    Thu thập URL sản phẩm từ tất cả URL danh mục.
    
    Args:
        max_load_more_clicks (int): Số lần tối đa nhấp nút "Xem thêm" cho mỗi danh mục
        
    Returns:
        list: Danh sách dict chứa URL sản phẩm và URL danh mục nguồn
    """
    category_urls = load_category_urls()
    if not category_urls:
        return []
    
    all_results = []
    driver = setup_driver()
    
    try:
        for index, category_url in enumerate(category_urls):
            print(f"\n--- Đang xử lý URL danh mục {index + 1}/{len(category_urls)} ---")
            
            product_links = crawl_products_from_category(driver, category_url, max_load_more_clicks)
            
            # Thêm vào kết quả
            for product_url in product_links:
                all_results.append({'URL': product_url, 'category_url': category_url})
                
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        driver.quit()
    
    return all_results

def save_product_urls(results):
    """
    Lưu kết quả vào file CSV.
    
    Args:
        results (list): Danh sách dict chứa thông tin URL sản phẩm.
    """
    if not results:
        print("Không có kết quả nào để lưu.")
        return
    
    try:
        df = pd.DataFrame(results)
        df.to_csv(PRODUCT_URLS_FILE, index=False)
        print(f"Đã lưu {len(results)} URL sản phẩm vào {PRODUCT_URLS_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu kết quả: {e}")

def main():
    """Hàm chính để thu thập và lưu URL sản phẩm."""
    max_load_more_clicks = 3  # Thay đổi số này để tải nhiều sản phẩm hơn
    results = crawl_product_urls(max_load_more_clicks)
    save_product_urls(results)

if __name__ == "__main__":
    main() 