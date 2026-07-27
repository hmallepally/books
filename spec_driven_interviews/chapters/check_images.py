import os
import re

chapters_dir = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'
focus_dirs = ['09-q1-implementation', '10-q2-matrix-simulation', '11-q3-hashmaps-sliding-windows', '12-q4-optimization-dp', '08-algorithms-assessment']

for d in focus_dirs:
    filepath = os.path.join(chapters_dir, d, 'base.md')
    if not os.path.exists(filepath): continue
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    images = re.findall(r'!\[.*?\]\((.*?)\)', content)
    for img in images:
        img_path = os.path.join(chapters_dir, d, img)
        if not os.path.exists(img_path):
            print(f'Missing image in {d}: {img}')
            
print("Done checking images.")
