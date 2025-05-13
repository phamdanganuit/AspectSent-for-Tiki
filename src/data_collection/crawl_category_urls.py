from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.constants import CATEGORY_URLS_FILE
from src.utils.driver_setup import setup_driver

def crawl_category_urls():
    """
    Thu thập URL của các danh mục sản phẩm từ trang chủ Tiki.
    
    Returns:
        list: Danh sách các URL danh mục đã thu thập được.
    """
    print("Bắt đầu thu thập URL danh mục từ Tiki...")
    
    url = "https://tiki.vn/"
    driver = setup_driver()
    urls_list = []
    
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        
        # Xử lý popup nếu có
        try:
            close_button_selector = "img[alt='close-icon']"
            close_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, close_button_selector))
            )
            close_button.click()
            print("Đã click nút đóng pop-up.")
        except TimeoutException:
            print("Không tìm thấy hoặc không thể click nút đóng pop-up.")
        except Exception as e:
            print(f"Lỗi khi click nút đóng: {e}")
        
        # Thu thập URL danh mục
        elements = driver.find_elements(
            By.XPATH,
            "//div[@class='sc-cffe1c5-0 bKBPyH'][div[text()='Danh mục']]//a"
        )
        
        hrefs = [el.get_attribute('href') for el in elements]
        for href in hrefs:
            if href and href not in urls_list:
                urls_list.append(href)
                
        print(f"Đã thu thập được {len(urls_list)} URL danh mục.")
        
    except Exception as e:
        print(f"Lỗi khi thu thập URL danh mục: {e}")
    finally:
        driver.quit()
    
    return urls_list

def save_category_urls(urls_list):
    """
    Lưu danh sách URL danh mục vào file CSV.
    
    Args:
        urls_list (list): Danh sách các URL danh mục cần lưu.
    """
    if not urls_list:
        print("Không có URL nào để lưu.")
        return
    
    try:
        df = pd.DataFrame(urls_list, columns=['URL'])
        df.to_csv(CATEGORY_URLS_FILE, index=False)
        print(f"Đã lưu danh sách URL vào {CATEGORY_URLS_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu danh sách URL: {e}")

def main():
    """Hàm chính để thu thập và lưu URL danh mục."""
    urls_list = crawl_category_urls()
    save_category_urls(urls_list)

if __name__ == "__main__":
    main() 