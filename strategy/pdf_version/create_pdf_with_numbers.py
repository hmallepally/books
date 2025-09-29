#!/usr/bin/env python3
"""
Add dummy page numbers to TOC and generate PDF using reportlab
"""

import os
import re

def add_dummy_page_numbers():
    """Add dummy page numbers to the Table of Contents"""
    
    # Read the markdown file
    with open('living_strategy_pdf_version.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add dummy page numbers to TOC entries
    # Pattern: - [Chapter X: Title](#link) - Description
    # Replace with: - [Chapter X: Title](#link) - Description ................. XX
    
    # Find all TOC entries and add page numbers
    toc_pattern = r'(- \[Chapter \d+: [^\]]+\]\([^)]+\) - [^\n]+)'
    
    def add_page_number(match):
        chapter_text = match.group(1)
        # Extract chapter number
        chapter_match = re.search(r'Chapter (\d+):', chapter_text)
        if chapter_match:
            chapter_num = int(chapter_match.group(1))
            # Calculate dummy page number (starting from page 10, each chapter ~3 pages)
            page_num = 10 + (chapter_num - 1) * 3
            return f"{chapter_text} ................. {page_num}"
        return chapter_text
    
    # Apply page numbers to TOC
    content = re.sub(toc_pattern, add_page_number, content)
    
    # Add page numbers to part headers in TOC
    part_pattern = r'(### Part [IVX]+: [^\n]+)'
    
    def add_part_page_number(match):
        part_text = match.group(1)
        # Extract part number
        if 'Part I:' in part_text:
            return f"{part_text} ................. 1"
        elif 'Part II:' in part_text:
            return f"{part_text} ................. 85"
        elif 'Part III:' in part_text:
            return f"{part_text} ................. 169"
        elif 'Part IV:' in part_text:
            return f"{part_text} ................. 253"
        elif 'Part V:' in part_text:
            return f"{part_text} ................. 337"
        elif 'Part VI:' in part_text:
            return f"{part_text} ................. 421"
        elif 'Part VII:' in part_text:
            return f"{part_text} ................. 505"
        return part_text
    
    content = re.sub(part_pattern, add_part_page_number, content)
    
    # Write updated content
    with open('living_strategy_pdf_version.md', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Added dummy page numbers to Table of Contents")

def create_pdf_with_reportlab():
    """Create PDF using reportlab library"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        
        # Create PDF
        doc = SimpleDocTemplate("Living_Strategy_PDF_Version.pdf", 
                              pagesize=letter,
                              rightMargin=1*inch,
                              leftMargin=1*inch,
                              topMargin=1*inch,
                              bottomMargin=1*inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        chapter_style = ParagraphStyle(
            'ChapterTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            spaceBefore=30,
            textColor=colors.darkblue
        )
        
        part_style = ParagraphStyle(
            'PartTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=40,
            spaceBefore=50,
            alignment=TA_CENTER,
            textColor=colors.darkred
        )
        
        toc_style = ParagraphStyle(
            'TOC',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=0
        )
        
        # Read markdown content
        with open('living_strategy_pdf_version.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into lines
        lines = content.split('\n')
        
        # Build PDF content
        story = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                story.append(Spacer(1, 6))
                continue
            
            # Handle different line types
            if line.startswith('# Living Strategy'):
                story.append(Paragraph(line[2:], title_style))
                story.append(Spacer(1, 20))
                
            elif line.startswith('## Table of Contents'):
                story.append(Paragraph(line[3:], chapter_style))
                story.append(Spacer(1, 20))
                
            elif line.startswith('### Part'):
                story.append(PageBreak())
                story.append(Paragraph(line[4:], part_style))
                story.append(Spacer(1, 20))
                
            elif line.startswith('# Part'):
                story.append(PageBreak())
                story.append(Paragraph(line[2:], part_style))
                story.append(Spacer(1, 20))
                
            elif line.startswith('# Chapter'):
                story.append(PageBreak())
                story.append(Paragraph(line[2:], chapter_style))
                story.append(Spacer(1, 20))
                
            elif line.startswith('## !['):
                # Skip image headers for now
                continue
                
            elif line.startswith('![Part'):
                # Handle part title images
                img_path = line.split('(')[1].split(')')[0]
                if os.path.exists(img_path):
                    try:
                        img = Image(img_path, width=6*inch, height=4*inch)
                        story.append(img)
                        story.append(Spacer(1, 20))
                    except:
                        pass
                        
            elif line.startswith('![Rocket') or line.startswith('![Lightbulb') or line.startswith('![Tools') or line.startswith('![Book') or line.startswith('![Chart'):
                # Skip emoji replacement images
                continue
                
            elif line.startswith('### '):
                story.append(Paragraph(line[4:], styles['Heading3']))
                story.append(Spacer(1, 12))
                
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], styles['Heading2']))
                story.append(Spacer(1, 12))
                
            elif line.startswith('- ['):
                # TOC entry
                story.append(Paragraph(line[2:], toc_style))
                
            elif line.startswith('> '):
                # Blockquote
                story.append(Paragraph(line[2:], styles['Normal']))
                story.append(Spacer(1, 6))
                
            elif line.startswith('**') and line.endswith('**'):
                # Bold text
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))
                
            else:
                # Regular paragraph
                if line:
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        print("✅ Successfully created Living_Strategy_PDF_Version.pdf")
        return True
        
    except ImportError:
        print("❌ reportlab not available. Installing...")
        try:
            import subprocess
            subprocess.check_call(['pip', 'install', 'reportlab'])
            print("✅ reportlab installed. Please run the script again.")
            return False
        except:
            print("❌ Could not install reportlab. Please install manually: pip install reportlab")
            return False
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        return False

def create_simple_pdf():
    """Create a simple PDF using basic HTML to PDF conversion"""
    try:
        import weasyprint
        
        # Read the HTML file
        with open('Living_Strategy_PDF_Version.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Convert to PDF
        weasyprint.HTML(string=html_content).write_pdf('Living_Strategy_PDF_Version.pdf')
        print("✅ Successfully created Living_Strategy_PDF_Version.pdf using WeasyPrint")
        return True
        
    except ImportError:
        print("❌ weasyprint not available")
        return False
    except Exception as e:
        print(f"❌ Error with WeasyPrint: {e}")
        return False

if __name__ == "__main__":
    print("Creating PDF version with dummy page numbers...")
    print("=" * 50)
    
    # Add dummy page numbers
    add_dummy_page_numbers()
    
    # Try different PDF creation methods
    success = False
    
    # Try reportlab first
    print("Trying reportlab...")
    success = create_pdf_with_reportlab()
    
    if not success:
        # Try weasyprint
        print("Trying weasyprint...")
        success = create_simple_pdf()
    
    if not success:
        print("❌ Could not create PDF automatically.")
        print("📄 Please use the HTML file and convert manually:")
        print("   1. Open Living_Strategy_PDF_Version.html in browser")
        print("   2. Print to PDF")
        print("   3. Adjust page settings as needed")
    
    print("=" * 50)
    print("✅ Dummy page numbers added to TOC")
    print("📄 You can now manually update the page numbers in the PDF")

