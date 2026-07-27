import os

f1_path = r'C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\15-mock-assessment-sets\base.md'
with open(f1_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Since the file is actually correctly formatted with newlines (except the weird line 38 stuff), wait, is it?
# Let's just do a direct string replacement on the exact text.

old_q4_8 = """* **Q4 (Hard): Union Find Network**
  * *Specification:* Find the redundant connection in a graph that should be a tree.
  * *Sample Test Case:* Input: `[[1,2],[1,3],[2,3]] -> [2,3]`
  * *Constraints:* Complexity bounds requiring optimal solution.
  * *Hint:* [PAT-17] Disjoint Set Union"""

new_q4_8 = """* **Q4 (Hard): Minimum Spanning Tree**
  * *Specification:* Given a weighted undirected graph, find the MST weight using Kruskal's algorithm with Union-Find.
  * *Sample Test Case:* Input: `edges -> weight`
  * *Constraints:* V \\le 10^4, E \\le 5 \\times 10^4.
  * *Hint:* [PAT-17] Disjoint Set Union + greedy edge sorting."""

# The problem was that the file literally contains '\n' instead of actual newlines.
# Let's replace literally "\n" if it's there.
old_q4_8_literal = r"* **Q4 (Hard): Union Find Network**\n  * *Specification:* Find the redundant connection in a graph that should be a tree.\n  * *Sample Test Case:* Input: `[[1,2],[1,3],[2,3]] -> [2,3]`\n  * *Constraints:* Complexity bounds requiring optimal solution.\n  * *Hint:* [PAT-17] Disjoint Set Union"

new_q4_8_literal = r"* **Q4 (Hard): Minimum Spanning Tree**\n  * *Specification:* Given a weighted undirected graph, find the MST weight using Kruskal's algorithm with Union-Find.\n  * *Sample Test Case:* Input: `edges -> weight`\n  * *Constraints:* V \le 10^4, E \le 5 \times 10^4.\n  * *Hint:* [PAT-17] Disjoint Set Union + greedy edge sorting."

if old_q4_8_literal in content:
    # It's literal!
    # But wait, there are two! Set 4 and Set 8. We need to replace only Set 8.
    # Set 8 comes after "## Set 8: "
    idx = content.find("## Set 8:")
    if idx != -1:
        next_idx = content.find(old_q4_8_literal, idx)
        if next_idx != -1:
            content = content[:next_idx] + new_q4_8_literal + content[next_idx+len(old_q4_8_literal):]

old_q4_14_literal = r"* **Q4 (Hard): Topological Sort Complex**\n  * *Specification:* Find the longest path in a Directed Acyclic Graph representing tasks.\n  * *Sample Test Case:* Input: `tasks -> 10 days`\n  * *Constraints:* Complexity bounds requiring optimal solution.\n  * *Hint:* [PAT-16] Topo Sort / DP"
new_q4_14_literal = r"* **Q4 (Hard): Course Schedule III**\n  * *Specification:* Given N courses with (duration, deadline), maximize courses completed.\n  * *Sample Test Case:* Input: `courses -> max`\n  * *Constraints:* N \le 10^4.\n  * *Hint:* [PAT-25] Priority Queue / Greedy with heap."

idx = content.find("## Set 14:")
if idx != -1:
    next_idx = content.find(old_q4_14_literal, idx)
    if next_idx != -1:
        content = content[:next_idx] + new_q4_14_literal + content[next_idx+len(old_q4_14_literal):]

old_q4_20_literal = r"* **Q4 (Hard): Dijkstra Shortest**\n  * *Specification:* Find network delay time for a signal to reach all nodes.\n  * *Sample Test Case:* Input: `nodes=4, edges -> 2`\n  * *Constraints:* Complexity bounds requiring optimal solution.\n  * *Hint:* [PAT-18] Dijkstra Priority Queue"
new_q4_20_literal = r"* **Q4 (Hard): Alien Dictionary**\n  * *Specification:* Given sorted alien words, derive character ordering.\n  * *Sample Test Case:* Input: `words -> ordering`\n  * *Constraints:* words \le 300, word length \le 100.\n  * *Hint:* Topological Sort on character graph."

idx = content.find("## Set 20:")
if idx != -1:
    next_idx = content.find(old_q4_20_literal, idx)
    if next_idx != -1:
        content = content[:next_idx] + new_q4_20_literal + content[next_idx+len(old_q4_20_literal):]

with open(f1_path, 'w', encoding='utf-8') as f:
    f.write(content)

