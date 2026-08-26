import os
import fitz
import re

def audit_pdf(pdf_path, lang_name):
    print(f"\n==================================================")
    print(f"   STARTING DEEP PDF AUDIT: {lang_name} ({pdf_path})")
    print(f"==================================================")
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found at {pdf_path}")
        return
        
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Total Pages: {total_pages}")
    
    issues = {
        'overflow_code': [],
        'double_numbering': [],
        'missing_images': [],
        'unresolved_refs': [],
        'orphan_headings': [],
        'large_gaps': [],
        'wrong_lang_syntax': []
    }
    
    # Language-specific keywords to check
    lang_checks = {
        'java': {'must_not_contain': ['def ', 'async def', 'using System;', 'namespace ', 'func ']},
        'python': {'must_not_contain': ['public class ', 'System.out.println', 'using System;', 'namespace ']},
        'csharp': {'must_not_contain': ['public class ', 'System.out.println', 'def ', 'async def']}
    }
    
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")
        lines = text.split('\n')
        
        # 1. Unresolved LaTeX refs or missing images
        if '??' in text or 'LaTeX Warning' in text:
            issues['unresolved_refs'].append((page_num + 1, "Unresolved reference (??)"))
        if '![' in text or '](visuals/' in text:
            issues['missing_images'].append((page_num + 1, "Un-rendered Markdown image tag found in text"))
            
        # 2. Check for double section numbering (e.g., "1.1.1 1.")
        double_num_match = re.search(r'\b(\d+\.\d+(\.\d+)?)\s+(\d+\.)\b', text)
        if double_num_match:
            issues['double_numbering'].append((page_num + 1, double_num_match.group(0)))
            
        # 3. Check for orphan headings near bottom of page
        rect = page.rect
        blocks = page.get_text("blocks")
        for b in blocks:
            # b: (x0, y0, x1, y1, text, block_no, block_type)
            block_text = b[4].strip()
            # If block is a heading (starts with number like 4.1 or Section) and y1 is near bottom (e.g. > 700 on 792 height page)
            if re.match(r'^\d+\.\d+\s+[A-Z]', block_text) and b[1] > 700:
                issues['orphan_headings'].append((page_num + 1, block_text[:40]))
                
        # 4. Check for code text overflowing right margin (x1 > 540)
        drawings = page.get_text("words")
        for w in drawings:
            # w: (x0, y0, x1, y1, word, block_no, line_no, word_no)
            if w[2] > 560: # 8.5in = 612pt. Margin = 1in (72pt) -> Right margin = 540pt
                if len(w[4]) > 5:
                    issues['overflow_code'].append((page_num + 1, w[4], round(w[2], 1)))
                    
        # 5. Wrong language syntax check
        check_rules = lang_checks.get(lang_name.lower(), {})
        for forbidden in check_rules.get('must_not_contain', []):
            if forbidden in text:
                # Exclude prologue or comparisons where other languages are explicitly discussed
                if page_num > 10 and 'Language Comparison' not in text:
                    issues['wrong_lang_syntax'].append((page_num + 1, f"Found '{forbidden}' in {lang_name} PDF"))
                    
    # Print Audit Summary
    print(f"\n--- AUDIT RESULTS FOR {lang_name} ---")
    print(f"1. Unrendered Images / Broken Tags: {len(issues['missing_images'])}")
    for p, detail in issues['missing_images']:
        print(f"   - Page {p}: {detail}")
        
    print(f"2. Double-Numbered Headings: {len(issues['double_numbering'])}")
    for p, detail in issues['double_numbering']:
        print(f"   - Page {p}: {detail}")
        
    print(f"3. Unresolved References (??): {len(issues['unresolved_refs'])}")
    for p, detail in issues['unresolved_refs']:
        print(f"   - Page {p}: {detail}")
        
    print(f"4. Orphan Headings at Page Bottom: {len(issues['orphan_headings'])}")
    for p, detail in issues['orphan_headings']:
        print(f"   - Page {p}: {detail}")
        
    print(f"5. Code / Word Right Margin Overflows (>560pt): {len(issues['overflow_code'])}")
    for p, word, x1 in issues['overflow_code'][:10]:
        print(f"   - Page {p}: '{word}' extended to x={x1}pt")
        
    print(f"6. Cross-Language Syntax Bleed: {len(issues['wrong_lang_syntax'])}")
    for p, detail in issues['wrong_lang_syntax'][:10]:
        print(f"   - Page {p}: {detail}")
        
    return issues

if __name__ == '__main__':
    audit_pdf('Java_Edition.pdf', 'Java')
    audit_pdf('Python_Edition.pdf', 'Python')
    audit_pdf('CSharp_Edition.pdf', 'CSharp')
