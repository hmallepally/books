```python
def ladder_length(self, begin_word: str, end_word: str, word_list: list[str]) -> int:
    word_set = set(word_list)
    if end_word not in word_set: return 0
    
    from collections import deque
    queue = deque([begin_word])
    level = 1
    
    while queue:
        for _ in range(len(queue)): # Level-by-level processing
            curr = queue.popleft()
            for j in range(len(curr)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == curr[j]: continue
                    next_word = curr[:j] + c + curr[j+1:]
                    if next_word == end_word: return level + 1
                    if next_word in word_set: # remove serves as 'visited' check
                        word_set.remove(next_word)
                        queue.append(next_word)
        level += 1
        
    return 0
# Time Complexity: O(M^2 * N) where M is word length, N is number of words
# Space Complexity: O(M * N)
```