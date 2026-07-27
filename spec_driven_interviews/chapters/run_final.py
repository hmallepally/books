import os

base_dir = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters'

def insert_after(file_path, search_str, text_to_insert):
    full_path = os.path.join(base_dir, file_path)
    if not os.path.exists(full_path):
        print('Missing file:', full_path)
        return
    with open(full_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if search_str in line:
            lines.insert(i+1, '\n' + text_to_insert + '\n')
            with open(full_path, 'w', encoding='utf-8') as fw:
                fw.writelines(lines)
            print('Inserted in', file_path)
            return
    print('Failed to find string in', file_path, '->', search_str)

# The missing 09 image
insert_after('09-algorithms-assessment/base.md', '| O(N!) |', '![Big-O Time Complexity Comparison Graph](visuals/big_o_comparison.jpg){width=85%}')

# The missing 16 image
insert_after('16-system-architecture/base.md', 'System design is a choice between **CP** and **AP**:', '![CAP Theorem — Consistency, Availability, and Partition Tolerance Trade-offs](visuals/cap_theorem.jpg){width=85%}')

# The constraint flowchart (user said 02-problem-decomposition/visuals/..., but text is in 09)
insert_after('09-algorithms-assessment/base.md', 'eliminates 50% of wrong algorithm choices', '![Constraint-to-Complexity Flowchart](../02-problem-decomposition/visuals/constraint_flowchart.jpg){width=85%}')

# And let's also try putting it in 02 after constraint analysis just in case
insert_after('02-problem-decomposition/base.md', 'We must solve this in $O(N)$ or $O(N \\log N)$ time.', '![Constraint-to-Complexity Flowchart](visuals/constraint_flowchart.jpg){width=85%}')

# The pacing strategy
insert_after('15-mock-assessment-sets/base.md', 'require further review.', '![Assessment Pacing Strategy and Time Allocation](visuals/pacing_strategy.jpg){width=85%}')

