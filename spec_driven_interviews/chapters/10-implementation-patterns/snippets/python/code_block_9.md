```python
def is_valid(self, s: str) -> bool:
    if not s or len(s) % 2 != 0:
        return False

    stack = []

    for c in s:
        if c == '(': stack.append(')')
        elif c == '{': stack.append('}')
        elif c == '[': stack.append(']')
        else:
            if not stack or stack.pop() != c:
                return False

    return len(stack) == 0 # Stack must be empty
# Time: O(N), Space: O(N) worst case for the stack
```