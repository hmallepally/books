import os

def patch_file(filepath, replacements):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    updated = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            updated += 1
    if updated > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully applied {updated} replacements to: {filepath}")
    else:
        print(f"No replacements matched in: {filepath}")

patch_file('chapters/11-matrix-grid-patterns/base.md', [
    ('can be done with two simpler operations', 'is achieved via two sequential operations'),
    ('Simply check every cell', 'Check every cell'),
    ('just count the "top-left" cell', 'count only the top-left cell')
])

patch_file('chapters/12-hashmaps-sliding-windows/base.md', [
    ('simple and reliable O(N²)', 'O(N²)'),
    ('heavily simplifies priority queue update patterns', 'eliminates priority queue update overhead'),
    ('Track required characters in a map. Expand right until all required characters are in the window, then contract left to minimize the window.',
     'Maintain a frequency map targetMap for string t and a dynamic window map windowMap. Track formed—the number of unique characters in t whose target frequency is met in the current window. Expand right until formed == targetMap.size(). Then contract left step-by-step to record the minimal valid window length, updating windowMap and decrementing formed when a required character count drops below target.'),
    ('Calculate the relative distance between adjacent characters. Use this sequence of differences as the HashMap key.',
     'Compute the normalized relative distance between adjacent characters using (s.charAt(i) - s.charAt(i-1) + 26) % 26. The resulting sequence of difference offsets forms a canonical HashMap key that groups all uniformly shifted strings together.'),
    ('Place number x at index x-1. Then scan to find the first index that doesn\'t have i+1.',
     'Use cyclic sort: while nums[i] > 0 and nums[i] <= N and nums[nums[i] - 1] != nums[i], swap nums[i] with nums[nums[i] - 1]. After at most N swaps across the entire array, scan from 0 to N-1; the first index where nums[i] != i + 1 identifies the smallest missing positive integer i + 1.'),
    ('Exactly(K) = AtMost(K) - AtMost(K-1).',
     'Counting subarrays with exactly K distinct elements directly using dynamic sliding window is difficult because contracting left can omit valid starting bounds non-monotonically. We compute exact K using cumulative bounds: Exactly(K) = AtMost(K) - AtMost(K-1), where atMost(X) uses a standard dynamic window.'),
    ('Record the direction moved (U, D, L, R) during DFS traversal. Store path strings in a HashSet to deduplicate identical shapes.',
     'Record the direction moved (\'U\', \'D\', \'L\', \'R\') during DFS traversal. Crucially, append a backtrack marker (e.g., \'B\') upon returning from each recursive call to prevent signature collisions between distinct island geometries. Store the resulting path strings in a HashSet.')
])

patch_file('chapters/13-optimization-dp/base.md', [
    ('The Key Trick: Think BACKWARDS', 'Core Strategy: Reverse Order Formulation (Last Burst Balloon)'),
    ('creates dependency chaos — bursting balloon `i` changes the neighbors of balloon `i+1`. Instead, ask: **"Which balloon do I burst LAST?"**',
     'introduces variable neighbor dependencies — bursting balloon `i` changes the adjacent neighbors of balloon `i+1`. Instead, determine **which balloon is burst LAST** in the interval `(i, j)`.'),
    ('use it as a warm-up before tackling the harder exemplars below',
     'use it as an introductory application of the pattern before tackling advanced exemplars below'),
    ('We maintain an array `tails` where `tails[i]` stores the smallest tail of all increasing subsequences of length `i+1`. We binary search the position to update in `tails`.',
     'Maintain an array `tails` where `tails[i]` stores the smallest tail value among all strictly increasing subsequences of length `i+1` found so far. The `tails` array is guaranteed to be strictly sorted. For each element `x` in `nums`, binary search for its insertion position in `tails`. If `x` is larger than all elements in `tails`, append it (extending the max LIS length by 1). Otherwise, replace the smallest tail >= x with `x`.')
])

patch_file('chapters/14-mastering-decomposition/base.md', [
    ('represent the apex of algorithmic assessments', 'represent the most complex assessment scenarios'),
    ('The synthesis secret here is **Reverse Thinking**:', 'The core analytical insight here is **Reverse Order Formulation**:'),
    ('The most fatal error.', 'The primary operational error.')
])
