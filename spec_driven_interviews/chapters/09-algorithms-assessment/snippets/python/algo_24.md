```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.is_word = True
        
    def search(self, word: str) -> bool:
        node = self._get_node(word)
        return node is not None and node.is_word
        
    def starts_with(self, prefix: str) -> bool:
        return self._get_node(prefix) is not None
        
    def _get_node(self, s: str) -> 'TrieNode':
        curr = self.root
        for c in s:
            if c not in curr.children:
                return None
            curr = curr.children[c]
        return curr
```
