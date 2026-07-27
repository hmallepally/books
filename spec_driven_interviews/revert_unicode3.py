import os
import subprocess
import re

chapters = [
    '00-prologue', '01-invariant-first', '02-problem-decomposition', '03-case-studies', 
    '04-oop-principles', '05-solid-boundaries', '06-functional-streams', '07-design-patterns', 
    '08-concurrency-performance', '09-algorithms-assessment', '11-matrix-grid-patterns', 
    '13-optimization-dp', '14-mastering-decomposition', '16-system-architecture', 
    '17-resiliency', '18-database-compliance', '19-behavioral-leadership', '20-testing-cicd', 
    '21-message-brokers', '22-aiml-llm', '23-appendix'
]

total_reverts = 0

for chapter in chapters:
    filepath = f"chapters/{chapter}/base.md"
    if not os.path.exists(filepath):
        continue
        
    try:
        # Use ./ relative to current directory for git show
        orig_content = subprocess.check_output(['git', 'show', f"6b2586b:./{filepath}"]).decode('utf-8')
    except subprocess.CalledProcessError as e:
        print("Error getting original for", filepath, e)
        continue
        
    with open(filepath, 'r', encoding='utf-8') as f:
        curr_lines = f.read().split('\n')
        
    orig_lines = orig_content.split('\n')
    
    def normalize(s):
        return re.sub(r'\W+', '', s)
        
    new_curr_lines = []
    
    for curr_line in curr_lines:
        if curr_line.strip().startswith('```'):
            new_curr_lines.append(curr_line)
            continue
            
        modified_line = curr_line
        
        for orig_line in orig_lines:
            if curr_line == orig_line:
                break
                
            test_line = curr_line
            
            if '> WARNING: **' in test_line and '> ⚠️ **' in orig_line:
                test_line = test_line.replace('> WARNING: **', '> ⚠️ **')
            elif 'WARNING:' in test_line and '⚠️' in orig_line:
                test_line = test_line.replace('WARNING:', '⚠️')
                
            if '$\\leq$' in test_line and '≤' in orig_line:
                test_line = test_line.replace('$\\leq$', '≤')
            if '$\\geq$' in test_line and '≥' in orig_line:
                test_line = test_line.replace('$\\geq$', '≥')
            if '$\\neq$' in test_line and '≠' in orig_line:
                test_line = test_line.replace('$\\neq$', '≠')
            if '$\\mu$' in test_line and 'μ' in orig_line:
                test_line = test_line.replace('$\\mu$', 'μ')
            if '$\\square$' in test_line and '∎' in orig_line:
                test_line = test_line.replace('$\\square$', '∎')
                
            if '> * **STAR' in test_line and '> ⭐ **STAR' in orig_line:
                test_line = test_line.replace('> * **STAR', '> ⭐ **STAR')
            elif '*' in test_line and '⭐' in orig_line:
                if normalize(curr_line) == normalize(orig_line):
                    test_line = orig_line
                    
            if test_line == orig_line or normalize(test_line) == normalize(orig_line):
                if curr_line != orig_line:
                    modified_line = orig_line
                    total_reverts += 1
                    print(f"Reverted in {chapter}:\n- {curr_line}\n+ {modified_line}")
                break
                
        new_curr_lines.append(modified_line)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_curr_lines))

print(f"Total reverted lines: {total_reverts}")
