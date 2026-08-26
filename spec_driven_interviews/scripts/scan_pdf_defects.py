import os
import glob
import re
import fitz

def scan_base_markdown_files():
    print("=== SCANNING BASE MARKDOWN FILES FOR FORMATTING BUGS ===")
    base_files = sorted(glob.glob('chapters/*/base.md'))
    
    hr_before_headings = []
    double_numbered_headings = []
    long_inline_code_in_lists = []
    
    for bf in base_files:
        ch_name = os.path.basename(os.path.dirname(bf))
        with open(bf, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for idx, line in enumerate(lines):
            line_str = line.strip()
            
            # 1. Check HR rules before headings
            if line_str in ['---', '***', '* * *']:
                next_idx = idx + 1
                while next_idx < len(lines) and lines[next_idx].strip() == '':
                    next_idx += 1
                if next_idx < len(lines) and lines[next_idx].strip().startswith('#'):
                    hr_before_headings.append((ch_name, idx + 1, lines[next_idx].strip()))
            
            # 2. Check heading double numbering e.g. ### 1. Title or ## 2. Title
            m = re.match(r'^(#{1,6})\s+(\d+\.?\s+)(.+)', line_str)
            if m:
                double_numbered_headings.append((ch_name, idx + 1, line_str))
            
            # 3. Check long inline code in list items
            if line_str.startswith('*') or line_str.startswith('-') or re.match(r'^\d+\.', line_str):
                code_blocks = re.findall(r'`([^`]{25,})`', line)
                if code_blocks:
                    for cb in code_blocks:
                        long_inline_code_in_lists.append((ch_name, idx + 1, line_str[:60], cb))
                        
    print(f"\n1. Found {len(hr_before_headings)} HR rules preceding section headings:")
    for ch, line_no, heading in hr_before_headings:
        print(f"   [{ch}:L{line_no}] {heading[:60]}")
        
    print(f"\n2. Found {len(double_numbered_headings)} Double-Numbered Headings:")
    for ch, line_no, heading in double_numbered_headings:
        print(f"   [{ch}:L{line_no}] {heading[:60]}")
        
    print(f"\n3. Found {len(long_inline_code_in_lists)} Long Inline Code Strings in Lists (causing justification gaps):")
    for ch, line_no, snippet, cb in long_inline_code_in_lists:
        print(f"   [{ch}:L{line_no}] `{cb}` in line: '{snippet}...'")
        
    return hr_before_headings, double_numbered_headings, long_inline_code_in_lists

if __name__ == '__main__':
    scan_base_markdown_files()
