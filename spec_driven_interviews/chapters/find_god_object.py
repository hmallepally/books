import os

base_dir = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.endswith('.md'):
            with open(os.path.join(root, f), 'r', encoding='utf-8') as file:
                content = file.read()
                if 'god object' in content.lower():
                    print(f'Found \"God Object\" in {os.path.join(root, f)}')
                if 'srp' in content.lower() or 'single responsibility' in content.lower():
                    print(f'Found \"SRP\" in {os.path.join(root, f)}')
