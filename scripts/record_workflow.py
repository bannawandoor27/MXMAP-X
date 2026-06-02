import sys
import time
import os
import urllib.request
from playwright.sync_api import sync_playwright

def check_server():
    print("Checking if FastAPI server is running on http://127.0.0.1:8000...")
    for _ in range(10):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/api/v1/health", timeout=2)
            print("Server is ready!")
            return True
        except Exception:
            time.sleep(1)
    return False

def record_workflow():
    if not check_server():
        print("ERROR: Could not connect to the FastAPI server.")
        print("Please start the server in a separate terminal using:")
        print("  make run")
        print("or")
        print("  uvicorn app.main:app --port 8000")
        sys.exit(1)
        
    output_dir = "workflow_video"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Launching browser and starting recording to {output_dir}...")
    try:
        with sync_playwright() as p:
            # Launch chromium in headed mode so it renders properly, but it will record automatically
            browser = p.chromium.launch(headless=True)
            
            # Create a context that records video
            context = browser.new_context(
                record_video_dir=output_dir,
                record_video_size={"width": 1280, "height": 720}
            )
            
            page = context.new_page()
            
            print("Navigating to Main Prediction page...")
            page.goto("http://127.0.0.1:8000/")
            page.wait_for_timeout(2000)
            
            # Click the 'Load Example Data' if present, or just Predict
            try:
                page.click("text='Load Example Data'", timeout=2000)
                page.wait_for_timeout(1000)
            except:
                pass
                
            page.click("button[type='submit']")
            try:
                page.wait_for_selector("text='Prediction Results'", timeout=10000)
            except:
                pass
            page.wait_for_timeout(3000)
            
            print("Navigating to Optimization page...")
            page.goto("http://127.0.0.1:8000/optimize")
            page.wait_for_timeout(2000)
            try:
                page.click("button:has-text('Run Optimization')", timeout=2000)
                page.wait_for_timeout(4000)
            except:
                pass
            
            print("Navigating to Exploration page...")
            page.goto("http://127.0.0.1:8000/explore")
            page.wait_for_timeout(2000)
            try:
                page.click("button:has-text('Generate Map')", timeout=2000)
                page.wait_for_timeout(4000)
            except:
                pass
            
            print("Navigating to AC-Line Filtering page...")
            page.goto("http://127.0.0.1:8000/filtering")
            page.wait_for_timeout(2000)
            try:
                page.click("button:has-text('Predict Performance')", timeout=2000)
                page.wait_for_timeout(4000)
            except:
                pass
            
            print("Navigating to Printing Process page...")
            page.goto("http://127.0.0.1:8000/printing")
            page.wait_for_timeout(2000)
            try:
                page.click("button:has-text('Get Recommendation')", timeout=2000)
                page.wait_for_timeout(4000)
            except:
                pass
            
            print("Navigating to Recipe Generator page...")
            page.goto("http://127.0.0.1:8000/recipes")
            page.wait_for_timeout(2000)
            try:
                page.click("button:has-text('High Capacitance')", timeout=2000)
                page.wait_for_timeout(2000)
            except:
                pass
            
            print("Closing browser (saving video)...")
            context.close()
            browser.close()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    print(f"Workflow recording completed! Video saved in: ./{output_dir}")

if __name__ == "__main__":
    try:
        import playwright
    except ImportError:
        print("Playwright is not installed. Please run:")
        print("pip install playwright")
        print("playwright install chromium")
        sys.exit(1)
        
    record_workflow()
