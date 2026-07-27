import os

base_dir = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'

def pr_match(fpath, kw):
    path = os.path.join(base_dir, fpath)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if kw.lower() in line.lower():
                print(f'{fpath} L{i+1}: {line.strip()}')

pr_match('09-algorithms-assessment/base.md', 'Max N for 1s')
pr_match('09-algorithms-assessment/base.md', 'O(N!)')
pr_match('16-system-architecture/base.md', 'Availability (A)')
pr_match('02-problem-decomposition/base.md', 'eliminates 50%')
pr_match('15-mock-assessment-sets/base.md', 'pacing')
pr_match('15-mock-assessment-sets/base.md', 'time management')
