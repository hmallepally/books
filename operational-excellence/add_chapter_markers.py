#!/usr/bin/env python3
"""
Add chapter end markers directly to the HTML file
"""

import re
import os

def add_chapter_end_markers_to_html():
    """Add creative chapter end markers to the HTML file"""
    
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print("🎨 Adding chapter end markers to HTML file...")
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add CSS for chapter markers
    css_addition = """
    <style>
    /* Chapter End Markers */
    .chapter-end-marker {
        text-align: center;
        margin: 40px 0 30px 0;
        padding: 20px 0;
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    .chapter-end-line {
        width: 60%;
        height: 3px;
        background: linear-gradient(90deg, #2c3e50, #3498db, #2c3e50);
        margin: 0 auto 15px auto;
        border-radius: 2px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chapter-end-dots {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        margin: 10px 0;
    }
    
    .chapter-end-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #3498db;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    
    .chapter-end-dot:nth-child(2) {
        background: #2c3e50;
        transform: scale(1.2);
    }
    
    .chapter-end-dot:nth-child(3) {
        background: #e74c3c;
    }
    
    .chapter-end-text {
        font-size: 10px;
        color: #7f8c8d;
        font-style: italic;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Print styles */
    @media print {
        .chapter-end-marker {
            margin: 30px 0 20px 0;
            padding: 15px 0;
        }
        
        .chapter-end-line {
            height: 2px;
        }
        
        .chapter-end-dot {
            width: 6px;
            height: 6px;
        }
        
        .chapter-end-text {
            font-size: 8px;
        }
    }
    </style>
    """
    
    # Insert CSS before closing head tag
    if '</head>' in content:
        content = content.replace('</head>', f'{css_addition}\n</head>')
    else:
        # If no head tag, add it
        content = f'<head>{css_addition}</head>\n{content}'
    
    # Find all chapter headings and add markers before them (except the first chapter)
    chapter_pattern = r'<h1 class="chapter-title">(Chapter \d+:.*?)</h1>'
    
    # Find all chapter positions
    chapters = list(re.finditer(chapter_pattern, content, re.IGNORECASE))
    
    print(f"Found {len(chapters)} chapters")
    
    # Add markers before chapters (starting from the second chapter)
    offset = 0
    for i, match in enumerate(chapters[1:], 1):  # Skip first chapter
        chapter_title = match.group(1)
        
        # Create chapter end marker HTML
        marker_html = f'''
        <div class="chapter-end-marker">
            <div class="chapter-end-line"></div>
            <div class="chapter-end-dots">
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
            </div>
            <div class="chapter-end-text">End of {chapter_title}</div>
        </div>
        '''
        
        # Insert marker before the chapter
        insert_pos = match.start() + offset
        content = content[:insert_pos] + marker_html + content[insert_pos:]
        offset += len(marker_html)
        
        print(f"✅ Added marker before: {chapter_title}")
    
    # Write the updated content
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n🎉 Chapter end markers added successfully!")
    print(f"📄 Updated file: {html_file}")
    
    return True

def main():
    """Main function"""
    print("🎨 Adding Chapter End Markers to HTML")
    print("=" * 50)
    
    success = add_chapter_end_markers_to_html()
    
    if success:
        print("\n✅ Chapter markers added successfully!")
        print("\n📋 Next steps:")
        print("   1. Generate a new PDF to see the chapter markers")
        print("   2. Review the visual improvements")
        print("   3. Check that page numbering is fixed")
    else:
        print("\n❌ Failed to add chapter markers. Please check the error messages above.")

if __name__ == "__main__":
    main()
