from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os
from .constants import BROWSER_PROFILES_DIR

def setup_driver(headless=True):
    """
    Khởi tạo và cấu hình trình duyệt Chrome với các tùy chọn phù hợp.
    
    Args:
        headless (bool): Chạy browser ở chế độ headless (không hiển thị giao diện)
    
    Returns:
        webdriver.Chrome: Instance trình duyệt đã được cấu hình.
    """
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    
    # Tùy chọn khác nhau cho chế độ headless và có giao diện
    if headless:
        # Cấu hình cho chế độ headless
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Không sử dụng profile người dùng khi headless
        print("Chạy Chrome ở chế độ headless không sử dụng profile")
    else:
        # Sử dụng chrome_data từ thư mục browser_profiles khi KHÔNG ở chế độ headless
        chrome_data_path = os.path.join(BROWSER_PROFILES_DIR, "chrome_data")
        if os.path.exists(chrome_data_path):
            chrome_options.add_argument(f"--user-data-dir={chrome_data_path}")
            print(f"Using Chrome profile from {chrome_data_path}")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver 