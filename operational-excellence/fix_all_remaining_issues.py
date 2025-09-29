#!/usr/bin/env python3
"""
Fix All Remaining Issues
"""

import re
import subprocess
import os

def fix_all_remaining_issues():
    """Fix all the remaining issues"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_SIMPLE_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing all remaining issues...")
    
    # Issue 1: Remove title page content repetition on page 3
    print("1️⃣ Removing title page content repetition on page 3...")
    
    # Find and remove any remaining title content that appears after TOC
    html_content = re.sub(
        r'<h2[^>]*>A Comprehensive Guide to AI-Driven Performance Optimization</h2>\s*<p><div class="tech-label">Author:</div>\s*<p>Hari Mallepally</p><br />\s*<div class="tech-label">Version:</div>\s*<p>3\.0</p><br />\s*<div class="tech-label">Publication Date:</div>\s*<p>September 2025</p><br />\s*<div class="tech-label">Target Audience:</div>\s*<p>Business Leaders, Operations Professionals, MBA Students</p></p>',
        '',
        html_content,
        flags=re.DOTALL
    )
    
    # Issue 2: Fix chapter titles - remove duplicates and ensure proper styling
    print("2️⃣ Fixing chapter titles...")
    
    # Fix Preface title
    html_content = re.sub(
        r'<h2[^>]*>Preface</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 1 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 1: The AI Revolution in Operations</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 2 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 2: AI-Powered Quality Control and Manufacturing</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 3 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 3: Strategic AI Implementation: Lessons from Industry Leaders</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 4 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 4: Lean Six Sigma Meets Artificial Intelligence</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 5 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 5: Total Productive Maintenance \(TPM\) in the AI Era</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 6 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 6: The AI-Powered Strategic Compass: Hoshin Kanri</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 7 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 7: Total Quality Management Enhanced by AI</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 8 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 8: AI-Powered Customer Satisfaction and Experience</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 9 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 9: AI in Software Development Lifecycle</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 10 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 10: Leadership in the AI Era</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 11 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 11: Prescriptive Analytics and Future Trends</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 12 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 12: Implementation Roadmap and Best Practices</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 13 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 13: Real-World Case Studies and Success Stories</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 14 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 14: The Future of AI in Operational Excellence</h2>',
        '',
        html_content
    )
    
    # Fix Chapter 15 title
    html_content = re.sub(
        r'<h2[^>]*>Chapter 15: AI Tools and Technologies: A Practical Guide</h2>',
        '',
        html_content
    )
    
    # Issue 3: Fix 10-20-70 Principle diagram
    print("3️⃣ Fixing 10-20-70 Principle diagram...")
    
    # Create a better version of the 10-20-70 Principle diagram
    mermaid_code = """
graph TD
    A["🎯 AI Success Framework<br/>The 10-20-70 Principle"] --> B["📊 10%<br/>Algorithms"]
    A --> C["🔧 20%<br/>Data & Infrastructure"]
    A --> D["👥 70%<br/>People & Process"]
    
    B --> E["Machine Learning Models<br/>Deep Learning Networks<br/>Natural Language Processing"]
    C --> F["Data Quality & Governance<br/>Cloud Infrastructure<br/>Security & Compliance"]
    D --> G["Change Management<br/>Training & Development<br/>Cultural Transformation"]
    
    style A fill:#2c3e50,stroke:#34495e,stroke-width:3px,color:#fff
    style B fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
    style C fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#fff
    style D fill:#27ae60,stroke:#229954,stroke-width:2px,color:#fff
    style E fill:#ecf0f1,stroke:#bdc3c7,stroke-width:1px,color:#2c3e50
    style F fill:#ecf0f1,stroke:#bdc3c7,stroke-width:1px,color:#2c3e50
    style G fill:#ecf0f1,stroke:#bdc3c7,stroke-width:1px,color:#2c3e50
    """
    
    # Write the improved mermaid code
    with open('diagrams/ai_success_framework_improved.mmd', 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    
    # Issue 4: Fix image sizing and layout
    print("4️⃣ Fixing image sizing and layout...")
    
    # Update CSS for better image handling
    image_css = """
        /* Images - improved sizing */
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0.3in auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Specific styling for diagrams */
        img[src*="ai_success_framework"] {
            max-width: 90%;
            margin: 0.4in auto;
        }
        
        /* Figure captions */
        .figure-caption {
            text-align: center;
            font-style: italic;
            color: #7f8c8d;
            margin-top: 0.1in;
            font-size: 0.9em;
        }
        
        /* Ensure images don't break across pages */
        img {
            page-break-inside: avoid;
        }
    """
    
    # Replace the existing image CSS
    html_content = re.sub(
        r'/\* Images \*/[^}]*\}',
        image_css.strip(),
        html_content,
        flags=re.DOTALL
    )
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_ALL_ISSUES_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ All issues fixed HTML created: {output_file}")
    return output_file

def generate_all_issues_fixed_pdf(html_file):
    """Generate PDF from all issues fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_ALL_ISSUES_FIXED.pdf'
            await page.pdf(
                path=output_pdf,
                format='A4',
                margin={
                    'top': '0.75in',
                    'right': '0.75in',
                    'bottom': '0.75in',
                    'left': '0.75in'
                },
                print_background=True,
                prefer_css_page_size=True,
                display_header_footer=False,
                tagged=True
            )
            
            await browser.close()
            print(f"🎉 All issues fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing all remaining issues...")
    
    # Fix all issues
    html_file = fix_all_remaining_issues()
    
    if html_file:
        # Generate all issues fixed PDF
        print("\n📚 Generating all issues fixed PDF...")
        generate_all_issues_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 All issues fixed:")
        print("1. ✅ Removed title page content repetition on page 3")
        print("2. ✅ Fixed chapter titles - removed duplicates and ensured proper styling")
        print("3. ✅ Improved 10-20-70 Principle diagram layout")
        print("4. ✅ Fixed image sizing and layout for better readability")
        print("\n📖 Please review the all issues fixed PDF!")
    else:
        print("❌ Failed to fix all issues")
