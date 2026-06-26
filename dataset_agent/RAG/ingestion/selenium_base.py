from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

class SeleniumScraper:
    """
    Base Selenium engine for rendering dynamic, JavaScript-heavy pages
    from scientific databases.
    """
    def __init__(self, headless: bool = True):
        options = Options()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("window-size=1920,1080")
        
        # Initialize Chrome driver
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.wait = WebDriverWait(self.driver, 15)
        print("Selenium Webdriver Initialized.")

    def get_page_source(self, url: str, wait_for_selector: str = None) -> BeautifulSoup:
        """
        Navigates to a URL, optionally waits for a specific CSS selector to load,
        and returns the BeautifulSoup parsed HTML.
        """
        print(f"Navigating to {url}...")
        self.driver.get(url)
        
        if wait_for_selector:
            try:
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector)))
            except Exception as e:
                print(f"Timeout waiting for {wait_for_selector}: {e}")
        else:
            # Brief implicit wait for generic JS to settle
            time.sleep(2)
            
        html = self.driver.page_source
        return BeautifulSoup(html, "html.parser")

    def close(self):
        self.driver.quit()
        print("Selenium Webdriver Closed.")

if __name__ == "__main__":
    # Test the scraper
    scraper = SeleniumScraper()
    soup = scraper.get_page_source("https://example.com")
    print("Extracted Title:", soup.title.string)
    scraper.close()
