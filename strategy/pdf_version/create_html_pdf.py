#!/usr/bin/env python3
"""
Create a simple HTML version that can be converted to PDF using browser print
"""

import os

def create_html_for_pdf():
    """Create HTML version optimized for PDF conversion"""
    
    # Read the markdown file
    with open('living_strategy_pdf_version.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert basic markdown to HTML
    html_content = convert_markdown_to_html(content)
    
    # Create HTML file
    with open('Living_Strategy_PDF_Version.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Created Living_Strategy_PDF_Version.html")
    print("📄 You can now:")
    print("   1. Open the HTML file in your browser")
    print("   2. Use Ctrl+P to print")
    print("   3. Choose 'Save as PDF'")
    print("   4. Set margins and page settings as needed")

def convert_markdown_to_html(content):
    """Basic markdown to HTML conversion"""
    
    # HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Living Strategy - PDF Version</title>
    <style>
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            page-break-before: always;
        }
        
        h1:first-of-type {
            page-break-before: auto;
        }
        
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        
        h3 {
            color: #7f8c8d;
            margin-top: 20px;
        }
        
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
            font-style: italic;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
        }
        
        ul, ol {
            margin: 10px 0;
            padding-left: 30px;
        }
        
        li {
            margin: 5px 0;
        }
        
        .toc {
            page-break-after: always;
        }
        
        .part-title {
            text-align: center;
            page-break-before: always;
            margin: 50px 0;
        }
        
        .part-title img {
            max-width: 80%;
            margin: 30px auto;
        }
        
        @media print {
            body {
                font-size: 12pt;
            }
            
            h1 {
                font-size: 18pt;
            }
            
            h2 {
                font-size: 16pt;
            }
            
            h3 {
                font-size: 14pt;
            }
        }
    </style>
</head>
<body>
{content}
</body>
</html>"""
    
    # Basic markdown conversions
    html = content
    
    # Convert headers
    html = html.replace('# ', '<h1>').replace('\n# ', '</h1>\n<h1>')
    html = html.replace('## ', '<h2>').replace('\n## ', '</h2>\n<h2>')
    html = html.replace('### ', '<h3>').replace('\n### ', '</h3>\n<h3>')
    
    # Convert images
    html = html.replace('![', '<img src="').replace('](', '" alt="').replace(')', '">')
    
    # Convert blockquotes
    html = html.replace('> ', '<blockquote>').replace('\n> ', '</blockquote>\n<blockquote>')
    
    # Convert lists
    html = html.replace('- ', '<li>').replace('\n- ', '</li>\n<li>')
    
    # Convert bold
    html = html.replace('**', '<strong>').replace('**', '</strong>')
    
    # Convert italic
    html = html.replace('*', '<em>').replace('*', '</em>')
    
    # Convert links
    html = html.replace('[', '<a href="').replace('](', '">').replace(')', '</a>')
    
    # Add closing tags for headers
    html = html.replace('<h1>', '</h1>\n<h1>')
    html = html.replace('<h2>', '</h2>\n<h2>')
    html = html.replace('<h3>', '</h3>\n<h3>')
    
    # Clean up
    html = html.replace('</h1>\n<h1>', '<h1>', 1)  # Remove first extra closing tag
    html = html.replace('</h2>\n<h2>', '<h2>', 1)
    html = html.replace('</h3>\n<h3>', '<h3>', 1)
    
    # Add closing tags at the end
    if html.endswith('<h1>'):
        html += '</h1>'
    elif html.endswith('<h2>'):
        html += '</h2>'
    elif html.endswith('<h3>'):
        html += '</h3>'
    
    # Wrap in template
    return html_template.replace('{content}', html)

if __name__ == "__main__":
    create_html_for_pdf()
