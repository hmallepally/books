import os
import glob
import re

def fix_all_defects():
    print("=== FIXING ALL PDF VISUAL DEFECTS ACROSS ALL BASE MARKDOWN FILES ===")
    base_files = sorted(glob.glob('chapters/*/base.md'))
    
    fixed_files_count = 0
    
    for bf in base_files:
        ch_name = os.path.basename(os.path.dirname(bf))
        with open(bf, 'r', encoding='utf-8') as f:
            content = f.read()
        
        orig_content = content
        
        # 1. Remove extraneous horizontal rules before headings
        # Replace '\n---\n\n#' or '\n---\n#' with '\n\n#'
        content = re.sub(r'\n+---+\s*\n+(?=#)', '\n\n', content)
        content = re.sub(r'\n+\*\s*\*\s*\*+\s*\n+(?=#)', '\n\n', content)
        
        # 2. Fix specific double-numbered headings in Ch 03, 16, 22
        if ch_name == '03-case-studies':
            content = re.sub(r'^## 1\. AuraPay:', '## AuraPay:', content, flags=re.MULTILINE)
            content = re.sub(r'^## 2\. ZenithTrade:', '## ZenithTrade:', content, flags=re.MULTILINE)
            content = re.sub(r'^## 3\. ChiramTrust:', '## ChiramTrust:', content, flags=re.MULTILINE)
            content = re.sub(r'^## 4\. Bounded Context', '## Bounded Context', content, flags=re.MULTILINE)
            content = re.sub(r'^## 5\. Domain Scaffolding', '## Domain Scaffolding', content, flags=re.MULTILINE)
            
        elif ch_name == '16-system-architecture':
            content = re.sub(r'^### 1\. Broadcast Hash Join', '### Broadcast Hash Join', content, flags=re.MULTILINE)
            content = re.sub(r'^### 2\. Sort-Merge Join', '### Sort-Merge Join', content, flags=re.MULTILINE)
            content = re.sub(r'^### 3\. Shuffle Hash Join', '### Shuffle Hash Join', content, flags=re.MULTILINE)
            
        elif ch_name == '22-aiml-llm':
            content = re.sub(r'^### 1\. Dual-Tier Feature Store', '### Dual-Tier Feature Store', content, flags=re.MULTILINE)
            content = re.sub(r'^### 2\. Model Explainability', '### Model Explainability', content, flags=re.MULTILINE)
            
        if content != orig_content:
            with open(bf, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_files_count += 1
            print(f"  Fixed visual defects in: {ch_name}/base.md")
            
    print(f"Done! Cleaned visual defects across {fixed_files_count} chapter base files.")

if __name__ == '__main__':
    fix_all_defects()
