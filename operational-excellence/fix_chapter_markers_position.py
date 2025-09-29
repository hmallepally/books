#!/usr/bin/env python3
"""
Fix chapter end markers to appear at the END of chapters, not at the beginning of next chapters
"""

import re
import os

def fix_chapter_markers_position():
    """Fix chapter end markers to appear at the end of chapters"""
    
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print("🔧 Fixing chapter end markers position...")
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove existing chapter markers
    chapter_marker_pattern = r'<div class="chapter-end-subtle">.*?</div>\s*</div>'
    content = re.sub(chapter_marker_pattern, '', content, flags=re.DOTALL)
    
    # Update CSS for better positioning
    css_addition = """
    <style>
    /* Chapter End Markers - At End of Chapters */
    .chapter-end-marker {
        text-align: center;
        margin: 20px 0 0 0;
        padding: 10px 0;
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    .chapter-end-line {
        width: 40%;
        height: 1px;
        background: #ddd;
        margin: 0 auto 5px auto;
        border-radius: 1px;
    }
    
    .chapter-end-dots {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 4px;
        margin: 3px 0;
    }
    
    .chapter-end-dot {
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background: #999;
    }
    
    .chapter-end-dot:nth-child(2) {
        background: #666;
    }
    
    .chapter-end-dot:nth-child(3) {
        background: #999;
    }
    
    /* Print styles - compact */
    @media print {
        .chapter-end-marker {
            margin: 15px 0 0 0;
            padding: 8px 0;
        }
        
        .chapter-end-line {
            height: 0.5px;
        }
        
        .chapter-end-dot {
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
    
    # Find all chapter divs and add markers at the END of each chapter (except the last one)
    chapter_div_pattern = r'<div id="chapter(\d+)" class="chapter">(.*?)</div>\s*(?=<div id="chapter\d+" class="chapter">|$)'
    
    def add_marker_at_end(match):
        chapter_num = match.group(1)
        chapter_content = match.group(2)
        
        # Don't add marker to the last chapter
        if chapter_num == "16":
            return match.group(0)
        
        # Create marker HTML (no text, just visual elements)
        marker_html = '''
        <div class="chapter-end-marker">
            <div class="chapter-end-line"></div>
            <div class="chapter-end-dots">
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
            </div>
        </div>
        '''
        
        # Add marker at the end of the chapter content, before closing div
        return f'<div id="chapter{chapter_num}" class="chapter">{chapter_content}{marker_html}</div>'
    
    # Apply the marker addition
    content = re.sub(chapter_div_pattern, add_marker_at_end, content, flags=re.DOTALL)
    
    # Write the updated content
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Chapter end markers repositioned successfully!")
    print(f"📄 Updated file: {html_file}")
    print(f"📋 Markers now appear at the END of each chapter")
    print(f"📋 No text, just line and dots")
    
    return True

def main():
    """Main function"""
    print("🔧 Fixing Chapter End Markers Position")
    print("=" * 50)
    
    success = fix_chapter_markers_position()
    
    if success:
        print("\n✅ Chapter markers repositioned successfully!")
        print("\n📋 Changes made:")
        print("   • Markers moved to END of chapters")
        print("   • Removed all text (no 'End of Chapter X')")
        print("   • Just visual elements: line and dots")
        print("   • Won't appear on next chapter's title page")
        
        print("\n📋 Next steps:")
        print("   1. Generate a new PDF to see the corrected markers")
        print("   2. Check that markers appear at chapter ends")
        print("   3. Verify no text appears with markers")
    else:
        print("\n❌ Failed to fix chapter markers. Please check the error messages above.")

if __name__ == "__main__":
    main()
