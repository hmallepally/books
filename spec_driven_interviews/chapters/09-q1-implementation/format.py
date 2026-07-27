import re

file_path = r'c:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\09-q1-implementation\base.md'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

new_lines = []

def is_problem_title(line):
    return bool(re.match(r'^\*\*\d+\.\s+.*?\*\*$', line))

def is_attribute(line):
    return bool(re.match(r'^\*\*.*?\*\*(:)?', line))

for i, line in enumerate(lines):
    if '## CodeSignal Classic Q1 Patterns' in line:
        continue
    
    line = line.replace('## Essential Terminology & Vocabulary', '## Section 1: Essential Terminology & Vocabulary')
    line = line.replace('## Reusable Code Templates', '## Section 2: Reusable Code Templates')
    line = line.replace('## Solved Exemplar Problems (20 Core Implementations)', '## Section 3: Solved Exemplar Problems')
    line = line.replace('## Practice Problem Bank (30 Detailed Problems with Hints)', '## Section 4: Practice Problem Bank')
    
    line = re.sub(r'^### Problem (\d+):\s*(.*?)$', r'**\1. \2**', line)
    line = re.sub(r'^\*\*Practice (\d+):\s*(.*?)\*\*$', r'**\1. \2**', line)
    
    line = line.replace('*Specification:*', '**Specification:**')
    line = line.replace('*Example:*', '**Example:**')
    line = line.replace('*Hint:*', '**Strategic Hint:**')
    
    new_lines.append(line)

compacted_lines = []
for i in range(len(new_lines)):
    line = new_lines[i]
    
    if line.strip() == '':
        prev_idx = len(compacted_lines) - 1
        while prev_idx >= 0 and compacted_lines[prev_idx].strip() == '':
            prev_idx -= 1
        
        next_idx = i + 1
        while next_idx < len(new_lines) and new_lines[next_idx].strip() == '':
            next_idx += 1
            
        if prev_idx >= 0 and next_idx < len(new_lines):
            prev_line = compacted_lines[prev_idx]
            next_line = new_lines[next_idx]
            
            if (is_problem_title(prev_line) or is_attribute(prev_line)) and is_attribute(next_line):
                continue
                
    compacted_lines.append(line)

final_lines = []
for i, line in enumerate(compacted_lines):
    if is_problem_title(line):
        prev_idx = len(final_lines) - 1
        while prev_idx >= 0 and final_lines[prev_idx].strip() == '':
            prev_idx -= 1
            
        if prev_idx >= 0:
            prev_non_empty = final_lines[prev_idx]
            if prev_non_empty != '* * *' and not prev_non_empty.startswith('## '):
                if final_lines[-1].strip() != '':
                    final_lines.append('')
                final_lines.append('* * *')
                final_lines.append('')
                
    final_lines.append(line)

with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(final_lines) + '\n')
