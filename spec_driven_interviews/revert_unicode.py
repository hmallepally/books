import subprocess
import os

chapters = [
    '00-prologue', '01-invariant-first', '02-problem-decomposition', '03-case-studies', 
    '04-oop-principles', '05-solid-boundaries', '06-functional-streams', '07-design-patterns', 
    '08-concurrency-performance', '09-algorithms-assessment', '11-matrix-grid-patterns', 
    '13-optimization-dp', '14-mastering-decomposition', '16-system-architecture', 
    '17-resiliency', '18-database-compliance', '19-behavioral-leadership', '20-testing-cicd', 
    '21-message-brokers', '22-aiml-llm', '23-appendix'
]

replacements_made = 0
for chapter in chapters:
    filepath = f'chapters/{chapter}/base.md'
    if not os.path.exists(filepath):
        continue
    
    diff_out = subprocess.check_output(['git', 'diff', '6b2586b', 'HEAD', '--', filepath]).decode('utf-8')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    new_content = content
    
    deleted_lines = []
    added_lines = []
    
    for diff_line in diff_out.split('\n'):
        if diff_line.startswith('-') and not diff_line.startswith('---'):
            deleted_lines.append(diff_line[1:])
        elif diff_line.startswith('+') and not diff_line.startswith('+++'):
            added_lines.append(diff_line[1:])
        elif not diff_line.startswith('-') and not diff_line.startswith('+'):
            if len(deleted_lines) == len(added_lines) and len(deleted_lines) > 0:
                for d, a in zip(deleted_lines, added_lines):
                    if any(c in d for c in ['⚠️', '≤', '≥', '≠', 'μ', '∎', '⭐']):
                        if a in new_content:
                            new_content = new_content.replace(a, d)
                            replacements_made += 1
            elif len(deleted_lines) > 0 or len(added_lines) > 0:
                # If they don't match in count, we might have a problem, but let's try to just find exact matches
                pass
            deleted_lines = []
            added_lines = []
            
    if len(deleted_lines) == len(added_lines) and len(deleted_lines) > 0:
        for d, a in zip(deleted_lines, added_lines):
            if any(c in d for c in ['⚠️', '≤', '≥', '≠', 'μ', '∎', '⭐']):
                if a in new_content:
                    new_content = new_content.replace(a, d)
                    replacements_made += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

print(f'Reverted {replacements_made} lines based on exact diff matches.')
