#!/usr/bin/env python3
"""
Remove chapter markers and fix page numbering issue
"""

import re
import os

def fix_page_numbers_only():
    """Remove chapter markers and fix page numbering"""
    
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print("🔧 Fixing page numbering and removing chapter markers...")
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove all chapter end markers
    chapter_marker_pattern = r'<div class="chapter-end-marker">.*?</div>\s*</div>'
    content = re.sub(chapter_marker_pattern, '', content, flags=re.DOTALL)
    
    # Remove duplicate CSS for chapter markers
    css_pattern = r'<style>\s*/\* Chapter End Markers \*/.*?</style>\s*'
    content = re.sub(css_pattern, '', content, flags=re.DOTALL)
    
    # Add simple CSS for better page breaks without extra spacing
    simple_css = """
    <style>
    /* Simple page break handling */
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
    
    ul, ol {
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    pre, code {
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    img {
        page-break-inside: avoid;
        break-inside: avoid;
        max-width: 100%;
        height: auto;
    }
    
    p {
        orphans: 2;
        widows: 2;
    }
    
    /* Print styles for better spacing */
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
    
    # Insert CSS before closing head tag
    if '</head>' in content:
        content = content.replace('</head>', f'{simple_css}\n</head>')
    else:
        content = f'<head>{simple_css}</head>\n{content}'
    
    # Write the updated content
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Page numbering fixed and chapter markers removed!")
    print(f"📄 Updated file: {html_file}")
    
    return True

def main():
    """Main function"""
    print("🔧 Fixing Page Numbers Only")
    print("=" * 40)
    
    success = fix_page_numbers_only()
    
    if success:
        print("\n✅ HTML file cleaned up successfully!")
        print("\n📋 Next steps:")
        print("   1. Generate a new PDF with fixed page numbering")
        print("   2. Should be back to ~195 pages")
        print("   3. No duplicate page numbers")
    else:
        print("\n❌ Failed to fix page numbers. Please check the error messages above.")

if __name__ == "__main__":
    main()
