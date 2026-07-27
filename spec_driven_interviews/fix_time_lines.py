import re

f = 'chapters/10-q2-matrix-simulation/base.md'
with open(f, 'r', encoding='utf-8') as fh:
    content = fh.read()

# Pattern: a line like '// Time: $...' followed by newline and '```'
# Move it OUTSIDE the code block
pattern = r'(// Time: )(\$[^$]+\$)( \| Space: )(\$[^$]+\$)\n```'

def replace_time_line(match):
    time_expr = match.group(2)
    space_expr = match.group(4)
    return '```\nTime: ' + time_expr + ' | Space: ' + space_expr

new_content = re.sub(pattern, replace_time_line, content)

count = len(re.findall(pattern, content))
print(f'Fixed {count} occurrences')

with open(f, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(new_content)
