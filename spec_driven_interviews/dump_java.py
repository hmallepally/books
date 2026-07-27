import os
import json

base = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'
for ch in ['10-implementation-patterns', '11-matrix-grid-patterns', '12-hashmaps-sliding-windows', '13-optimization-dp']:
    d = os.path.join(base, ch, 'snippets', 'java')
    blocks = {}
    files = [f for f in os.listdir(d) if f.startswith('code_block_') and f.endswith('.md')]
    for f in sorted(files, key=lambda x: int(x.split('_')[2].split('.')[0])):
        with open(os.path.join(d, f), 'r') as fp:
            blocks[f] = fp.read()
    with open(f'{ch}_java.json', 'w') as fp:
        json.dump(blocks, fp, indent=2)
