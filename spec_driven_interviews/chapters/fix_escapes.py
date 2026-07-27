import os

f1_path = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\15-mock-assessment-sets\base.md'
with open(f1_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the over-escaped backslashes
content = content.replace(r"\\\\n", r"\n")
content = content.replace(r"\\le", r"\le")
content = content.replace(r"\\times", r"\times")
content = content.replace(r"\\", r"") # wait, let's just do it directly

with open(f1_path, 'w', encoding='utf-8') as f:
    f.write(content)
