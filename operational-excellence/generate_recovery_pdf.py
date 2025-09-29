#!/usr/bin/env python3
"""
Generate PDF from the recovered HTML file
"""

from playwright.sync_api import sync_playwright
import os

def generate_pdf():
    """Generate PDF from the recovered HTML file"""
    
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    output_file = "final_versions/with_toc/Operational_Excellence_with_AI_RECOVERED.pdf"
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print("🔄 Generating PDF from recovered HTML file...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Load the HTML file
        page.goto(f"file://{os.path.abspath(html_file)}")
        
        # Wait for content to load
        page.wait_for_load_state('networkidle')
        
        # Generate PDF with KDP specifications
        page.pdf(
            path=output_file,
            format='Letter',  # 8.5" x 11" for KDP
            margin={
                'top': '0.75in',
                'right': '0.75in',
                'bottom': '0.75in',
                'left': '0.75in'
            },
            print_background=True,
            prefer_css_page_size=True
        )
        
        browser.close()
    
    print(f"✅ PDF generated successfully: {output_file}")
    return True

if __name__ == "__main__":
    generate_pdf()
