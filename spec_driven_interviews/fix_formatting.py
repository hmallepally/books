"""
Fix formatting across chapters 9-12:
1. Add blank lines between Specification/Example/Pattern/Explanation fields
   so Pandoc renders them as separate paragraphs.
2. In practice problems: Example on own paragraph, Constraints inline OK.
"""
import re

chapters = [
    "chapters/09-q1-implementation/base.md",
    "chapters/10-q2-matrix-simulation/base.md",
    "chapters/11-q3-hashmaps-sliding-windows/base.md",
    "chapters/12-q4-optimization-dp/base.md",
]

# Fields that should start new paragraphs (blank line before them)
# We want: Specification, Example, Pattern, Explanation each as own paragraph
# Also handle custom labels from Ch9 like **Why two passes?**, **Critical edge case:**, etc.

for ch_path in chapters:
    print(f"\nProcessing: {ch_path}")
    
    with open(ch_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    changes = 0
    
    for i, line in enumerate(lines):
        stripped = line.rstrip('\n')
        
        # Check if this line starts with a field label that needs a blank line before it
        needs_blank = False
        
        if stripped.startswith('**Example:**') or stripped.startswith('**Example**:'):
            needs_blank = True
        elif stripped.startswith('**Pattern:**') or stripped.startswith('**Pattern**:'):
            needs_blank = True
        elif stripped.startswith('**Explanation:**') or stripped.startswith('**Explanation**:'):
            needs_blank = True
        # Ch9 custom labels
        elif stripped.startswith('**Why ') and stripped.endswith('**'):
            needs_blank = True
        elif stripped.startswith('**Critical edge case:**'):
            needs_blank = True
        elif stripped.startswith('**Common mistake:**'):
            needs_blank = True
        elif stripped.startswith('**Why not '):
            needs_blank = True
        elif stripped.startswith('**Invariant:**'):
            needs_blank = True
        
        # Only add blank line if previous line is NOT already blank
        if needs_blank and i > 0:
            prev = lines[i-1].rstrip('\n')
            if prev != '' and not prev.startswith('* * *') and not prev.startswith('---'):
                new_lines.append('\n')
                changes += 1
        
        new_lines.append(line)
    
    print(f"  Added {changes} blank lines for paragraph breaks")
    
    with open(ch_path, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(new_lines)
    
    print(f"  Saved!")

print("\nAll chapters processed!")
