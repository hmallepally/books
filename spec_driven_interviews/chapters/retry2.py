import os
base_dir = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'
def pr_match(fpath, kw):
    path = os.path.join(base_dir, fpath)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if kw.lower() in line.lower():
                    print(f'{fpath} L{i+1}: {line.strip()}')

pr_match('02-problem-decomposition/base.md', 'complexity')
pr_match('02-problem-decomposition/base.md', 'rule')
pr_match('15-mock-assessment-sets/base.md', 'pacing')
pr_match('15-mock-assessment-sets/base.md', 'time')
