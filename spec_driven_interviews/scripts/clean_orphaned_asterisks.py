import os

def clean_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    cleaned_lines = []
    removed_count = 0
    
    for line in lines:
        if line.strip() in ['*', '* ']:
            removed_count += 1
            continue
        cleaned_lines.append(line)
        
    if removed_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        print(f"Removed {removed_count} stray asterisks from: {file_path}")

def clean_all(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file in ['base.md', 'chapter.md']:
                clean_file(os.path.join(root, file))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_all(os.path.join(base_dir, 'chapters'))
    clean_all(os.path.join(base_dir, 'editions'))
