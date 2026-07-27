import os

base_dir = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'
files = {
    '01-invariant-first': ['binary search'],
    '02-problem-decomposition': ['sub-problem', 'sub-problems', 'constraint', 'complexity'],
    '03-case-studies': ['ZenithTrade', 'ChiramTrust'],
    '04-oop-principles': ['God Object', 'SRP', 'Single Responsibility', 'violation'],
    '06-functional-streams': ['lazy', 'short-circuit'],
    '08-concurrency-performance': ['deadlock', 'states', 'lifecycle', 'thread'],
    '09-algorithms-assessment': ['Big-O', 'complexity'],
    '14-mastering-decomposition': ['Canvas', 'pattern selection', 'pattern'],
    '15-mock-assessment-sets': ['pacing'],
    '16-system-architecture': ['CAP theorem', 'monolith', 'microservices', 'consistent hashing', 'CAP'],
    '18-database-compliance': ['sharding'],
    '21-message-brokers': ['partition', 'consumer group'],
    '22-aiml-llm': ['serving', 'model']
}
out = {}
for d, terms in files.items():
    path = os.path.join(base_dir, d, 'base.md')
    out[d] = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            for term in terms:
                if term.lower() in line.lower():
                    out[d].append(f'L{i+1}: {line.strip()}')
                    break

with open('search_out.txt', 'w', encoding='utf-8') as f:
    for d, lines in out.items():
        f.write(f'\n--- {d}/base.md ---\n')
        for line in lines:
            f.write(f'{line}\n')
