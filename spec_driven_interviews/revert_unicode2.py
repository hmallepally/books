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

replacements = [
    ('WARNING:', '⚠️'),
    ('$\\leq$', '≤'),
    ('$\\geq$', '≥'),
    ('$\\neq$', '≠'),
    ('$\\mu$', 'μ'),
    ('$\\square$', '∎'),
    ('*', '⭐')
]

total_reverts = 0

for chapter in chapters:
    filepath = f"chapters/{chapter}/base.md"
    if not os.path.exists(filepath):
        continue
        
    try:
        orig_content = subprocess.check_output(['git', 'show', f"6b2586b:{filepath}"]).decode('utf-8')
    except subprocess.CalledProcessError:
        continue
        
    with open(filepath, 'r', encoding='utf-8') as f:
        curr_lines = f.read().split('\n')
        
    orig_lines = orig_content.split('\n')
    
    # We will do a fuzzy match. For each line in curr_lines, we see if there is a highly similar line in orig_lines
    # that contains the unicode char, where the current line contains the ascii replacement.
    
    # Helper to strip all non-alphanumeric chars for loose comparison
    def normalize(s):
        return re.sub(r'\W+', '', s)
        
    new_curr_lines = []
    
    for curr_line in curr_lines:
        if curr_line.strip().startswith('```'):
            new_curr_lines.append(curr_line)
            continue
            
        modified_line = curr_line
        
        # Check against all original lines to see if there's a match
        for orig_line in orig_lines:
            # If they are exactly the same, no need to revert
            if curr_line == orig_line:
                break
                
            # Check if this orig_line is the unicode version of the curr_line
            # We can replace the ascii with unicode in curr_line and see if it matches orig_line
            
            test_line = curr_line
            
            # 1. > WARNING: ** -> > ⚠️ **
            if '> WARNING: **' in test_line and '> ⚠️ **' in orig_line:
                test_line = test_line.replace('> WARNING: **', '> ⚠️ **')
            elif 'WARNING:' in test_line and '⚠️' in orig_line:
                test_line = test_line.replace('WARNING:', '⚠️')
                
            # 2-6. Math and symbols
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
                
            # 7. ⭐ -> *
            if '> * **STAR' in test_line and '> ⭐ **STAR' in orig_line:
                test_line = test_line.replace('> * **STAR', '> ⭐ **STAR')
            elif '*' in test_line and '⭐' in orig_line:
                # Be careful not to replace bolding stars. Only replace a single * that corresponds to ⭐.
                # Actually, let's just do a direct character replace for the one that differs.
                # If removing all non-alphanumerics makes them match:
                if normalize(curr_line) == normalize(orig_line):
                    test_line = orig_line # Just take the original line!
                    
            if test_line == orig_line or normalize(test_line) == normalize(orig_line):
                # The orig_line is the correct reverted version of the text
                # But wait, what if the new line had a valid other change? 
                # Since the instruction says visual image references were added (on new lines)
                # and we don't want to lose those. The visual images are standalone lines `![...]`
                # So if normalize(test_line) == normalize(orig_line), the only difference is symbols.
                if curr_line != orig_line:
                    modified_line = orig_line
                    total_reverts += 1
                break
                
        new_curr_lines.append(modified_line)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_curr_lines))

print(f"Total reverted lines: {total_reverts}")
