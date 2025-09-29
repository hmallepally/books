#!/usr/bin/env python3
"""
Add very subtle chapter end markers that won't increase page count
"""

import re
import os

def add_subtle_chapter_markers():
    """Add very subtle chapter end markers"""
    
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print("🎨 Adding subtle chapter end markers...")
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add very minimal CSS for chapter markers
    css_addition = """
    <style>
    /* Very Subtle Chapter End Markers */
    .chapter-end-subtle {
        text-align: center;
        margin: 15px 0 10px 0;
        padding: 8px 0;
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    .chapter-end-line-subtle {
        width: 40%;
        height: 1px;
        background: #ddd;
        margin: 0 auto 5px auto;
        border-radius: 1px;
    }
    
    .chapter-end-dots-subtle {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 4px;
        margin: 3px 0;
    }
    
    .chapter-end-dot-subtle {
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background: #999;
    }
    
    .chapter-end-dot-subtle:nth-child(2) {
        background: #666;
    }
    
    .chapter-end-dot-subtle:nth-child(3) {
        background: #999;
    }
    
    /* Print styles - even more compact */
    @media print {
        .chapter-end-subtle {
            margin: 10px 0 5px 0;
            padding: 5px 0;
        }
        
        .chapter-end-line-subtle {
            height: 0.5px;
        }
        
        .chapter-end-dot-subtle {
            width: 3px;
            height: 3px;
        }
    }
    </style>
    """
    
    # Insert CSS before closing head tag
    if '</head>' in content:
        content = content.replace('</head>', f'{css_addition}\n</head>')
    else:
        content = f'<head>{css_addition}</head>\n{content}'
    
    # Find all chapter headings and add subtle markers before them (except the first chapter)
    chapter_pattern = r'<h1 class="chapter-title">(Chapter \d+:.*?)</h1>'
    
    # Find all chapter positions
    chapters = list(re.finditer(chapter_pattern, content, re.IGNORECASE))
    
    print(f"Found {len(chapters)} chapters")
    
    # Add subtle markers before chapters (starting from the second chapter)
    offset = 0
    for i, match in enumerate(chapters[1:], 1):  # Skip first chapter
        chapter_title = match.group(1)
        
        # Create very subtle chapter end marker HTML
        marker_html = f'''
        <div class="chapter-end-subtle">
            <div class="chapter-end-line-subtle"></div>
            <div class="chapter-end-dots-subtle">
                <div class="chapter-end-dot-subtle"></div>
                <div class="chapter-end-dot-subtle"></div>
                <div class="chapter-end-dot-subtle"></div>
            </div>
        </div>
        '''
        
        # Insert marker before the chapter
        insert_pos = match.start() + offset
        content = content[:insert_pos] + marker_html + content[insert_pos:]
        offset += len(marker_html)
        
        print(f"✅ Added subtle marker before: {chapter_title}")
    
    # Write the updated content
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n🎉 Subtle chapter end markers added successfully!")
    print(f"📄 Updated file: {html_file}")
    
    return True

def main():
    """Main function"""
    print("🎨 Adding Subtle Chapter End Markers")
    print("=" * 50)
    
    success = add_subtle_chapter_markers()
    
    if success:
        print("\n✅ Subtle chapter markers added successfully!")
        print("\n📋 Next steps:")
        print("   1. Generate a new PDF to see the subtle markers")
        print("   2. Check if page count remains ~195")
        print("   3. Review the visual improvements")
    else:
        print("\n❌ Failed to add chapter markers. Please check the error messages above.")

if __name__ == "__main__":
    main()
