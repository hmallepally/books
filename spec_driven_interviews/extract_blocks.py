import os
import re

chapters = [
    '10-implementation-patterns',
    '11-matrix-grid-patterns',
    '12-hashmaps-sliding-windows',
    '13-optimization-dp'
]

base_dir = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'

for ch in chapters:
    ch_dir = os.path.join(base_dir, ch)
    base_md = os.path.join(ch_dir, 'base.md')
    
    if not os.path.exists(base_md):
        print(f'Missing {base_md}')
        continue
        
    snippets_dir = os.path.join(ch_dir, 'snippets')
    os.makedirs(os.path.join(snippets_dir, 'java'), exist_ok=True)
    os.makedirs(os.path.join(snippets_dir, 'python'), exist_ok=True)
    os.makedirs(os.path.join(snippets_dir, 'csharp'), exist_ok=True)
    
    with open(base_md, 'r', encoding='utf-8') as f:
        content = f.read()
        
    pattern = re.compile(r'```java\n(.*?)```\n?', re.DOTALL)
    
    matches = pattern.findall(content)
    print(f'{ch}: found {len(matches)} java blocks')
    
    counter = 0
    def repl(m):
        global counter
        counter += 1
        block = m.group(1)
        filename = f'code_block_{counter}.md'
        
        with open(os.path.join(snippets_dir, 'java', filename), 'w', encoding='utf-8') as sf:
            sf.write('```java\n' + block + '```\n')
            
        return f"{{{{ inject('{filename}') }}}}"
        
    new_content = pattern.sub(repl, content)
    
    with open(base_md, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
print('Extraction complete.')
