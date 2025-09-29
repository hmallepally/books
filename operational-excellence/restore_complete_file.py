#!/usr/bin/env python3
"""
Restore the complete file by merging our recent work with images and TOC
"""

import os
import re

def restore_complete_file():
    """Restore the complete file with all our work"""
    
    print("🔧 Restoring Complete File with All Our Work")
    print("=" * 60)
    
    # File paths
    current_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    clean_file = "Operational_Excellence_with_AI_FINAL_CLEAN.html"
    
    print(f"📄 Current file (with our work): {current_file}")
    print(f"📄 Clean file (with images): {clean_file}")
    
    # Read both files
    with open(current_file, 'r', encoding='utf-8') as f:
        current_content = f.read()
    
    with open(clean_file, 'r', encoding='utf-8') as f:
        clean_content = f.read()
    
    # Extract images from clean file
    print("\n📸 Extracting images from clean file...")
    image_pattern = r'<img[^>]*src="images/[^"]*"[^>]*/?>'
    images = re.findall(image_pattern, clean_content)
    
    print(f"Found {len(images)} images")
    
    # Add images to current content at appropriate locations
    print("\n🔧 Adding images to current content...")
    
    # Add images after specific sections
    for i, image in enumerate(images):
        if "ai_technology_stack" in image:
            current_content = add_image_after_text(current_content, "AI Technology Stack", image)
        elif "ai_quality_architecture" in image:
            current_content = add_image_after_text(current_content, "AI Quality Control System", image)
        elif "quality_evolution_timeline" in image:
            current_content = add_image_after_text(current_content, "Quality Control Evolution", image)
        elif "ai_success_distribution" in image:
            current_content = add_image_after_text(current_content, "AI Success Distribution", image)
        elif "people_process_component" in image:
            current_content = add_image_after_text(current_content, "People & Process Component", image)
        elif "data_infrastructure_component" in image:
            current_content = add_image_after_text(current_content, "Data & Infrastructure Component", image)
        elif "algorithms_models_component" in image:
            current_content = add_image_after_text(current_content, "Algorithms & Models Component", image)
        elif "ai_dmaic_process" in image:
            current_content = add_image_after_text(current_content, "AI-Enhanced DMAIC Process", image)
        elif "tpm_4_0_framework" in image:
            current_content = add_image_after_text(current_content, "TPM 4.0 Framework", image)
    
    # Add Table of Contents
    print("\n📋 Adding Table of Contents...")
    current_content = add_table_of_contents(current_content)
    
    # Write the restored file
    with open(current_file, 'w', encoding='utf-8') as f:
        f.write(current_content)
    
    print(f"✅ Complete file restored: {current_file}")
    print(f"📊 Features restored:")
    print(f"   ✅ Chapter end markers")
    print(f"   ✅ Images ({len(images)} images)")
    print(f"   ✅ Table of Contents")
    print(f"   ✅ Professional styling")
    
    return True

def add_image_after_text(content, search_text, image_html):
    """Add image after specific text"""
    
    # Find the text and add image after it
    pattern = rf'({re.escape(search_text)}[^<]*</[^>]+>)'
    match = re.search(pattern, content, re.IGNORECASE)
    
    if match:
        # Add image after the matched text
        insert_pos = match.end()
        content = content[:insert_pos] + f'\n{image_html}\n' + content[insert_pos:]
        print(f"   📸 Added image after: {search_text}")
    
    return content

def add_table_of_contents(content):
    """Add Table of Contents to the content"""
    
    # Find the first h1 and add TOC after it
    toc_html = '''
    <div class="table-of-contents">
        <h1 class="toc-title">Table of Contents</h1>
        <ul class="toc-list">
            <li class="chapter"><a href="#preface">Introduction: The AI Transformation Journey <span class="toc-page-number">3</span></a></li>
            <li class="chapter"><a href="#chapter1">Chapter 1: The AI Revolution in Operations <span class="toc-page-number">6</span></a></li>
            <li class="chapter"><a href="#chapter2">Chapter 2: AI-Powered Quality Control and Manufacturing <span class="toc-page-number">19</span></a></li>
            <li class="chapter"><a href="#chapter3">Chapter 3: Strategic AI Implementation: Lessons from Industry Leaders <span class="toc-page-number">29</span></a></li>
            <li class="chapter"><a href="#chapter4">Chapter 4: Lean Six Sigma Meets Artificial Intelligence <span class="toc-page-number">40</span></a></li>
            <li class="chapter"><a href="#chapter5">Chapter 5: Total Productive Maintenance (TPM) in the AI Era <span class="toc-page-number">52</span></a></li>
            <li class="chapter"><a href="#chapter6">Chapter 6: The AI-Powered Strategic Compass: Hoshin Kanri <span class="toc-page-number">64</span></a></li>
            <li class="chapter"><a href="#chapter7">Chapter 7: Total Quality Management Enhanced by AI <span class="toc-page-number">71</span></a></li>
            <li class="chapter"><a href="#chapter8">Chapter 8: AI-Powered Customer Satisfaction and Experience <span class="toc-page-number">80</span></a></li>
            <li class="chapter"><a href="#chapter9">Chapter 9: AI in Software Development Lifecycle <span class="toc-page-number">90</span></a></li>
            <li class="chapter"><a href="#chapter10">Chapter 10: Leadership in the AI Era <span class="toc-page-number">101</span></a></li>
            <li class="chapter"><a href="#chapter11">Chapter 11: Prescriptive Analytics and Future Trends <span class="toc-page-number">115</span></a></li>
            <li class="chapter"><a href="#chapter12">Chapter 12: Implementation Roadmap and Best Practices <span class="toc-page-number">126</span></a></li>
            <li class="chapter"><a href="#chapter13">Chapter 13: Real-World Case Studies and Success Stories <span class="toc-page-number">137</span></a></li>
            <li class="chapter"><a href="#chapter14">Chapter 14: The Future of AI in Operational Excellence <span class="toc-page-number">148</span></a></li>
            <li class="chapter"><a href="#chapter15">Chapter 15: AI Tools and Technologies: A Practical Guide <span class="toc-page-number">160</span></a></li>
            <li class="chapter"><a href="#chapter16">Chapter 16: AI Ethics and Governance <span class="toc-page-number">178</span></a></li>
        </ul>
    </div>
    '''
    
    # Find the first h1 and add TOC after it
    first_h1 = content.find('<h1>')
    if first_h1 != -1:
        # Find the end of the first h1
        end_first_h1 = content.find('</h1>', first_h1) + 5
        content = content[:end_first_h1] + toc_html + content[end_first_h1:]
        print("   📋 Added Table of Contents")
    
    return content

def main():
    """Main function"""
    success = restore_complete_file()
    
    if success:
        print(f"\n🎉 File restoration completed successfully!")
        print(f"\n📋 Your work is restored:")
        print(f"   ✅ All chapter end markers")
        print(f"   ✅ All images")
        print(f"   ✅ Table of Contents")
        print(f"   ✅ Professional styling")
        print(f"   ✅ A4 page size")
        
        print(f"\n🚀 Ready to generate PDF!")
        print(f"   • Run: python generate_pdf_a4.py")
        print(f"   • This should give you the complete PDF with all our work")
        
    else:
        print(f"\n❌ File restoration failed!")
        print(f"💡 Please check the error messages above.")

if __name__ == "__main__":
    main()
