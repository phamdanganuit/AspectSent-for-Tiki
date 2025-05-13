import pandas as pd
import time
import sys
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.constants import BRONZE_DIR
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

def crawl_product_reviews(url):
    """
    Thu thập đánh giá từ một URL sản phẩm.
    
    Args:
        url (str): URL sản phẩm cần thu thập đánh giá
        
    Returns:
        list: Danh sách đánh giá thu thập được
    """
    if not url:
        print("URL không hợp lệ")
        return []
    
    all_reviews = []
    driver = setup_driver()
    
    try:
        print(f"\n--- Đang xử lý URL sản phẩm: {url} ---")
        
        reviews = navigate_through_reviews(driver, url)
        if reviews:
            # Thêm URL sản phẩm vào mỗi đánh giá
            for review in reviews:
                review['product_url'] = url
            
            all_reviews.extend(reviews)
            print(f"Đã thu thập được {len(all_reviews)} đánh giá từ URL: {url}")
                
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        driver.quit()
    
    return all_reviews

if __name__ == "__main__":
    # Ví dụ sử dụng
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://tiki.vn/smart-tivi-lg-4k-43-inch-43uq7550psf-uhd-webos-chinh-hang-p192950037.html"
    
    reviews = crawl_product_reviews(url)
    print(f"Kết quả: thu thập được {len(reviews)} đánh giá")
    
    # Chỉ hiển thị 3 đánh giá đầu tiên để xem mẫu
    if reviews:
        print("\nMẫu đánh giá thu được:")
        for i, review in enumerate(reviews[:3]):
            print(f"\nĐánh giá {i+1}:")
            print(f"Tiêu đề: {review['title']}")
            print(f"Nội dung: {review['content'][:100]}...")
    else:
        print("Không thu thập được đánh giá nào.") 