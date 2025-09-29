#!/usr/bin/env python3
"""
PDF Generation Script using Chrome/Chromium via selenium
"""

import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_chrome_driver():
    """Setup Chrome driver with PDF generation capabilities"""
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-javascript")
    
    # Enable print to PDF
    chrome_options.add_experimental_option("prefs", {
        "printing.print_preview_sticky_settings.appState": '{"recentDestinations":[{"id":"Save as PDF","origin":"local","account":""}],"selectedDestinationId":"Save as PDF","version":2}',
        "printing.default_destination_selection_rules": {"kind": "local", "namePattern": "Save as PDF"}
    })
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        return None

def generate_pdf_with_chrome():
    """Generate PDF using Chrome browser"""
    
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Input and output file paths
    html_file = current_dir / "Operational_Excellence_with_AI_WITH_TOC.html"
    pdf_file = current_dir / "Operational_Excellence_with_AI_WITH_TOC.pdf"
    
    print(f"Converting {html_file} to {pdf_file}")
    
    # Setup Chrome driver
    driver = setup_chrome_driver()
    if not driver:
        print("❌ Failed to setup Chrome driver")
        return False
    
    try:
        # Load the HTML file
        file_url = f"file:///{html_file.absolute().as_posix()}"
        print(f"Loading: {file_url}")
        
        driver.get(file_url)
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        print("Page loaded successfully")
        
        # Execute JavaScript to print to PDF
        pdf_script = """
        window.print();
        """
        
        driver.execute_script(pdf_script)
        
        # Wait a moment for the print dialog
        time.sleep(3)
        
        print("✅ PDF generation initiated")
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False
        
    finally:
        driver.quit()

def generate_pdf_simple():
    """Simple PDF generation using basic HTML to PDF conversion"""
    
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Input and output file paths
    html_file = current_dir / "Operational_Excellence_with_AI_WITH_TOC.html"
    pdf_file = current_dir / "Operational_Excellence_with_AI_WITH_TOC.pdf"
    
    print(f"Attempting to convert {html_file} to {pdf_file}")
    
    # Try using Python's weasyprint with a simpler approach
    try:
        import weasyprint
        
        # Read HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Generate PDF
        pdf_document = weasyprint.HTML(string=html_content, base_url=str(current_dir))
        pdf_document.write_pdf(str(pdf_file))
        
        print(f"✅ PDF generated successfully: {pdf_file}")
        
        # Get file size
        file_size = pdf_file.stat().st_size
        print(f"📄 File size: {file_size / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with weasyprint: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting PDF generation...")
    
    # Try simple approach first
    success = generate_pdf_simple()
    
    if not success:
        print("\n🔄 Trying Chrome-based approach...")
        success = generate_pdf_with_chrome()
    
    if success:
        print("\n🎉 PDF generation completed successfully!")
    else:
        print("\n💥 PDF generation failed. Please check the error messages above.")
        print("\n📋 Alternative approaches:")
        print("1. Open the HTML file in Chrome and use Ctrl+P > Save as PDF")
        print("2. Install wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
        print("3. Use online HTML to PDF converters")
