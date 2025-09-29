import pdfplumber
import re
from pathlib import Path

def extract_pdf_content(pdf_path):
    """Extract text content from PDF and structure it"""
    content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                content.append(f"<!-- Page {page_num} -->\n{text}\n")
    
    return '\n'.join(content)

def structure_content_to_html(text_content):
    """Convert extracted text to HTML structure"""
    lines = text_content.split('\n')
    html_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip page markers
        if line.startswith('<!-- Page'):
            continue
            
        # Detect chapter titles (they usually start with "Chapter X:" or "Introduction:")
        if re.match(r'^(Chapter \d+:|Introduction:)', line):
            html_lines.append(f'<h1 class="chapter-title">{line}</h1>')
        # Detect section headings (usually in caps or title case)
        elif re.match(r'^[A-Z][A-Z\s]+$', line) and len(line) > 10:
            html_lines.append(f'<h2>{line}</h2>')
        # Regular paragraphs
        else:
            html_lines.append(f'<p>{line}</p>')
    
    return '\n'.join(html_lines)

def create_html_from_pdf(pdf_path, output_path):
    """Main function to create HTML from PDF"""
    print(f"Extracting content from {pdf_path}...")
    
    # Extract text content
    text_content = extract_pdf_content(pdf_path)
    
    # Save raw extracted text for review
    with open('extracted_text.txt', 'w', encoding='utf-8') as f:
        f.write(text_content)
    print("Raw extracted text saved to extracted_text.txt")
    
    # Structure into HTML
    html_content = structure_content_to_html(text_content)
    
    # Create full HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Operational Excellence with AI</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .chapter-title {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        @page {{
            size: A4;
            margin: 2cm;
        }}
    </style>
</head>
<body>
    <h1>Operational Excellence with AI</h1>
    <h2>Table of Contents</h2>
    <!-- TOC will need to be manually added -->
    
    {html_content}
</body>
</html>"""
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"HTML file created: {output_path}")
    print("Note: This is a basic reconstruction. You may need to manually:")
    print("1. Add proper table of contents")
    print("2. Format tables properly")
    print("3. Add images")
    print("4. Adjust styling")

if __name__ == "__main__":
    pdf_path = "final_versions/with_toc/Operational_Excellence_with_AI_COMPLETE.pdf"
    output_path = "final_versions/with_toc/Operational_Excellence_with_AI_RECONSTRUCTED.html"
    
    create_html_from_pdf(pdf_path, output_path)
