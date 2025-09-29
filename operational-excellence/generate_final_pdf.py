#!/usr/bin/env python3
"""
Generate final PDF with enhanced page break handling and table optimization
"""

import os
import subprocess
import sys
from pathlib import Path

def add_page_break_css():
    """Add CSS to prevent orphaned headings and improve page breaks"""
    css_addition = """
    <style>
    /* Enhanced page break handling */
    h1, h2, h3, h4, h5, h6 {
        page-break-after: avoid;
        break-after: avoid;
        orphans: 3;
        widows: 3;
    }
    
    /* Keep headings with their content */
    h1 + *, h2 + *, h3 + *, h4 + *, h5 + *, h6 + * {
        page-break-before: avoid;
        break-before: avoid;
    }
    
    /* Table page break handling */
    table {
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    /* Large tables can break but keep header */
    table.comparison-table,
    table.implementation-table,
    table.roadmap-table {
        page-break-inside: auto;
        break-inside: auto;
    }
    
    table.comparison-table thead,
    table.implementation-table thead,
    table.roadmap-table thead {
        page-break-after: avoid;
        break-after: avoid;
        display: table-header-group;
    }
    
    /* Keep table rows together when possible */
    tr {
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    /* List handling */
    ul, ol {
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    /* Code blocks */
    pre, code {
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    /* Images */
    img {
        page-break-inside: avoid;
        break-inside: avoid;
        max-width: 100%;
        height: auto;
    }
    
    /* Paragraph spacing */
    p {
        orphans: 2;
        widows: 2;
    }
    
    /* Chapter breaks */
    .chapter {
        page-break-before: always;
        break-before: always;
    }
    
    /* Better spacing for print */
    @media print {
        body {
            font-size: 11pt;
            line-height: 1.4;
        }
        
        h1 { font-size: 18pt; margin-top: 0; }
        h2 { font-size: 16pt; margin-top: 12pt; }
        h3 { font-size: 14pt; margin-top: 10pt; }
        h4 { font-size: 12pt; margin-top: 8pt; }
        
        table {
            font-size: 10pt;
        }
        
        .comparison-table th,
        .implementation-table th,
        .roadmap-table th {
            font-size: 10pt;
            padding: 4pt;
        }
        
        .comparison-table td,
        .implementation-table td,
        .roadmap-table td {
            font-size: 9pt;
            padding: 3pt;
        }
    }
    </style>
    """
    return css_addition

def enhance_html_for_pdf(html_file):
    """Enhance HTML file with better page break handling"""
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add enhanced CSS for page breaks
    css_addition = add_page_break_css()
    
    # Insert CSS before closing head tag
    if '</head>' in content:
        content = content.replace('</head>', f'{css_addition}\n</head>')
    else:
        # If no head tag, add it
        content = f'<head>{css_addition}</head>\n{content}'
    
    # Create enhanced version
    enhanced_file = html_file.replace('.html', '_ENHANCED_FOR_PDF.html')
    with open(enhanced_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return enhanced_file

def generate_pdf_with_wkhtmltopdf(html_file, output_file):
    """Generate PDF using wkhtmltopdf with optimized settings"""
    
    cmd = [
        "wkhtmltopdf",
        "--page-size", "A4",
        "--margin-top", "0.75in",
        "--margin-right", "0.75in", 
        "--margin-bottom", "0.75in",
        "--margin-left", "0.75in",
        "--encoding", "UTF-8",
        "--enable-local-file-access",
        "--print-media-type",
        "--disable-smart-shrinking",
        "--zoom", "1.0",
        "--dpi", "300",
        "--javascript-delay", "2000",  # Wait for content to load
        "--no-stop-slow-scripts",
        "--enable-javascript",
        "--disable-external-links",
        "--disable-internal-links",
        "--footer-center", "[page]",
        "--footer-font-size", "8",
        "--footer-spacing", "5",
        html_file,
        output_file
    ]
    
    print("Running wkhtmltopdf command...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ PDF generated successfully: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        return True
    else:
        print(f"❌ wkhtmltopdf error:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

def generate_pdf_with_weasyprint(html_file, output_file):
    """Generate PDF using WeasyPrint"""
    try:
        import weasyprint
        
        print("Using WeasyPrint for PDF generation...")
        
        # Read HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Generate PDF with enhanced settings
        html_doc = weasyprint.HTML(
            string=html_content, 
            base_url=os.path.dirname(os.path.abspath(html_file))
        )
        
        # Generate PDF with custom CSS for better page breaks
        css = weasyprint.CSS(string="""
            @page {
                size: A4;
                margin: 0.75in;
            }
            
            h1, h2, h3, h4, h5, h6 {
                page-break-after: avoid;
                break-after: avoid;
                orphans: 3;
                widows: 3;
            }
            
            h1 + *, h2 + *, h3 + *, h4 + *, h5 + *, h6 + * {
                page-break-before: avoid;
                break-before: avoid;
            }
            
            table {
                page-break-inside: avoid;
                break-inside: avoid;
            }
            
            table.comparison-table,
            table.implementation-table,
            table.roadmap-table {
                page-break-inside: auto;
                break-inside: auto;
            }
            
            table.comparison-table thead,
            table.implementation-table thead,
            table.roadmap-table thead {
                page-break-after: avoid;
                break-after: avoid;
                display: table-header-group;
            }
            
            tr {
                page-break-inside: avoid;
                break-inside: avoid;
            }
        """)
        
        html_doc.write_pdf(output_file, stylesheets=[css])
        
        print(f"✅ PDF generated successfully with WeasyPrint: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        return True
        
    except ImportError:
        print("❌ WeasyPrint not available")
        return False
    except Exception as e:
        print(f"❌ WeasyPrint error: {e}")
        return False

def generate_pdf_with_playwright(html_file, output_file):
    """Generate PDF using Playwright"""
    try:
        import asyncio
        from playwright.async_api import async_playwright
        
        async def generate():
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                # Load HTML file
                await page.goto(f"file://{os.path.abspath(html_file)}")
                
                # Wait for content to load
                await page.wait_for_load_state('networkidle')
                
                # Generate PDF with enhanced settings
                await page.pdf(
                    path=output_file,
                    format='A4',
                    margin={
                        'top': '0.75in',
                        'right': '0.75in',
                        'bottom': '0.75in',
                        'left': '0.75in'
                    },
                    print_background=True,
                    prefer_css_page_size=True,
                    display_header_footer=True,
                    header_template='<div></div>',
                    footer_template='<div style="font-size: 8px; text-align: center; width: 100%;"><span class="pageNumber"></span></div>'
                )
                
                await browser.close()
        
        print("Using Playwright for PDF generation...")
        asyncio.run(generate())
        
        print(f"✅ PDF generated successfully with Playwright: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        return True
        
    except ImportError:
        print("❌ Playwright not available")
        return False
    except Exception as e:
        print(f"❌ Playwright error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Generating Enhanced PDF with Better Page Break Handling")
    print("=" * 70)
    
    # Define paths
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    output_file = "final_versions/with_toc/Operational_Excellence_with_AI_ENHANCED.pdf"
    
    # Check if HTML file exists
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print(f"📄 Input HTML: {html_file}")
    print(f"📄 Output PDF: {output_file}")
    
    # Enhance HTML for better PDF generation
    print("\n🔧 Enhancing HTML for better page break handling...")
    enhanced_html = enhance_html_for_pdf(html_file)
    print(f"✅ Enhanced HTML created: {enhanced_html}")
    
    # Try different PDF generation methods
    success = False
    
    # Method 1: wkhtmltopdf (best for tables)
    print("\n📊 Method 1: Trying wkhtmltopdf...")
    try:
        success = generate_pdf_with_wkhtmltopdf(enhanced_html, output_file)
        if success:
            print("✅ wkhtmltopdf succeeded!")
    except FileNotFoundError:
        print("⚠️  wkhtmltopdf not found, trying alternatives...")
    
    # Method 2: WeasyPrint (if wkhtmltopdf failed)
    if not success:
        print("\n📊 Method 2: Trying WeasyPrint...")
        success = generate_pdf_with_weasyprint(enhanced_html, output_file)
        if success:
            print("✅ WeasyPrint succeeded!")
    
    # Method 3: Playwright (if others failed)
    if not success:
        print("\n📊 Method 3: Trying Playwright...")
        success = generate_pdf_with_playwright(enhanced_html, output_file)
        if success:
            print("✅ Playwright succeeded!")
    
    # Clean up enhanced HTML file
    if os.path.exists(enhanced_html):
        os.remove(enhanced_html)
        print(f"🧹 Cleaned up temporary file: {enhanced_html}")
    
    if success:
        print(f"\n🎉 PDF generation completed successfully!")
        print(f"📄 Final PDF: {output_file}")
        print(f"📊 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        print("\n📋 Key improvements in this PDF:")
        print("   • Better page break handling for headings and tables")
        print("   • Reduced orphaned headings")
        print("   • Improved table rendering")
        print("   • Enhanced spacing and typography")
        print("   • Professional page numbering")
        
        print("\n🔍 Please review the PDF and check:")
        print("   • Section headings stay with their content")
        print("   • Tables render properly without page breaks")
        print("   • Overall layout and readability")
        
    else:
        print("\n❌ All PDF generation methods failed!")
        print("\n💡 To install required tools:")
        print("   • wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
        print("   • pip install weasyprint")
        print("   • pip install playwright && playwright install")
        print("   • pip install pdfkit")

if __name__ == "__main__":
    main()
