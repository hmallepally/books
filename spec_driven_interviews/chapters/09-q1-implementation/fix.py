import re

file_path = r'c:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\09-q1-implementation\base.md'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

for i in range(len(lines)):
    lines[i] = re.sub(r'\*+Specification:\*+', '**Specification:**', lines[i])
    lines[i] = re.sub(r'\*+Example:\*+', '**Example:**', lines[i])
    lines[i] = re.sub(r'\*+Strategic Hint:\*+', '**Strategic Hint:**', lines[i])
    lines[i] = re.sub(r'\*+Hint:\*+', '**Strategic Hint:**', lines[i])
    # Also if there's a space after colon, e.g. **Specification:**
    
with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines) + '\n')
