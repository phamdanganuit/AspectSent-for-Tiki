from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os
from .constants import BROWSER_PROFILES_DIR

def setup_driver():
    """
    Khởi tạo và cấu hình trình duyệt Chrome với các tùy chọn phù hợp.
    
    Returns:
        webdriver.Chrome: Instance trình duyệt đã được cấu hình.
    """
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    
    # Sử dụng chrome_data từ thư mục browser_profiles nếu có
    chrome_data_path = os.path.join(BROWSER_PROFILES_DIR, "chrome_data")
    if os.path.exists(chrome_data_path):
        chrome_options.add_argument(f"--user-data-dir={chrome_data_path}")
        print(f"Using Chrome profile from {chrome_data_path}")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver 