#!/usr/bin/env python3
"""
Add chapter end marker for Introduction and fix page numbering duplicates
"""

import re
import os

def add_intro_marker_and_fix_pages():
    """Add marker for Introduction and fix page numbering"""
    
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print("🔧 Adding Introduction marker and fixing page numbering...")
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update CSS to fix page numbering duplicates
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
    
    /* Fix page numbering duplicates */
    @media print {
        /* Hide any duplicate page numbers */
        .page-number-duplicate {
            display: none !important;
        }
        
        /* Ensure single page numbering */
        @page {
            @bottom-center {
                content: counter(page);
                font-size: 8px;
                color: #666;
            }
        }
        
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
    
    # Add marker for Introduction section
    intro_pattern = r'(<div id="preface" class="chapter">.*?</div>)\s*(?=<div id="chapter1" class="chapter">)'
    
    def add_intro_marker(match):
        intro_content = match.group(1)
        
        # Create marker HTML
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
        
        # Add marker at the end of introduction
        return f'{intro_content}{marker_html}'
    
    # Apply the intro marker
    content = re.sub(intro_pattern, add_intro_marker, content, flags=re.DOTALL)
    
    # Write the updated content
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Introduction marker added and page numbering fixed!")
    print(f"📄 Updated file: {html_file}")
    print(f"📋 Added marker for Introduction section")
    print(f"📋 Fixed page numbering duplicates")
    
    return True

def main():
    """Main function"""
    print("🔧 Adding Introduction Marker & Fixing Page Numbers")
    print("=" * 60)
    
    success = add_intro_marker_and_fix_pages()
    
    if success:
        print("\n✅ Changes completed successfully!")
        print("\n📋 Changes made:")
        print("   • Added chapter end marker for Introduction")
        print("   • Fixed page numbering duplicates")
        print("   • CSS improvements for print")
        
        print("\n📋 Next steps:")
        print("   1. Generate a new PDF to see the changes")
        print("   2. Check that Introduction has end marker")
        print("   3. Verify single page numbers (no duplicates)")
    else:
        print("\n❌ Failed to make changes. Please check the error messages above.")

if __name__ == "__main__":
    main()
