from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:8501")
        # wait for app to load completely
        time.sleep(10)
        page.screenshot(path="docs/assets/dashboard_preview.png", full_page=True)
        browser.close()

if __name__ == '__main__':
    run()
