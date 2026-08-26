import os
import re

KEYWORDS = [
    r'\*\*Initialization:\*\*',
    r'\*\*Maintenance:\*\*',
    r'\*\*Termination:\*\*',
    r'\*\*Definition:\*\*',
    r'\*\*Why it matters:\*\*',
    r'\*\*When to use:\*\*',
    r'\*\*Why it matters on Easy-tier:\*\*',
    r'\*\*Trade-off:\*\*',
    r'\*\*Interview Rule:\*\*',
    r'\*\*Crucial Mistake:\*\*',
    r'\*\*Restatement:\*\*',
    r'\*\*Pattern Mapping:\*\*',
    r'\*\*Approach:\*\*',
    r'\*\*Inputs:\*\*',
    r'\*\*Outputs:\*\*',
    r'\*\*Constraints:\*\*',
    r'\*\*Edge Cases:\*\*',
    r'\*\*Sub-Problems:\*\*',
    r'\*\*Complexity Target:\*\*'
]

def fix_content(content):
    # 1. Separate inline keyword labels (e.g. "... true. **Maintenance:** ...") onto a fresh line
    for kw in KEYWORDS:
        # If keyword is preceded by non-newline text (e.g. "something. **Maintenance:**")
        pattern = r'([^\n])\s+(' + kw + r')'
        content = re.sub(pattern, r'\1\n\n\2', content)

    lines = content.split('\n')
    fixed_lines = []
    
    list_num_re = re.compile(r'^\s*\d+\.\s+')
    list_bullet_re = re.compile(r'^\s*[\-\*]\s+')
    kw_start_re = re.compile(r'^\s*(' + '|'.join(KEYWORDS) + r')')

    for i, line in enumerate(lines):
        prev_line = fixed_lines[-1] if fixed_lines else ""
        
        # Check if line starts a numbered list
        if list_num_re.match(line):
            # If prev_line is not empty and not already a list item, insert blank line
            if prev_line.strip() != "" and not list_num_re.match(prev_line) and not list_bullet_re.match(prev_line) and not prev_line.strip().startswith("#"):
                fixed_lines.append("")
        
        # Check if line starts a bullet list
        elif list_bullet_re.match(line):
            if prev_line.strip() != "" and not list_bullet_re.match(prev_line) and not list_num_re.match(prev_line) and not prev_line.strip().startswith("#"):
                fixed_lines.append("")

        # Check if line starts with one of our key labels
        elif kw_start_re.match(line):
            if prev_line.strip() != "":
                fixed_lines.append("")

        fixed_lines.append(line)
        
    return '\n'.join(fixed_lines)

def process_chapters(chapters_dir):
    count = 0
    for root, dirs, files in os.walk(chapters_dir):
        for file in files:
            if file in ['base.md', 'chapter.md']:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    original = f.read()
                
                fixed = fix_content(original)
                if original != fixed:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed)
                    print(f"Fixed formatting in: {file_path}")
                    count += 1
    print(f"Total files updated: {count}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    process_chapters(os.path.join(base_dir, 'chapters'))
    process_chapters(os.path.join(base_dir, 'editions'))
