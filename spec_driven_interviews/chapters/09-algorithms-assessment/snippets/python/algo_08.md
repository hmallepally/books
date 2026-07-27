```python
def is_valid_parentheses(s: str) -> bool:
    stack = []
    for c in s:
        if c == '(':
            stack.append(')')
        elif c == '{':
            stack.append('}')
        elif c == '[':
            stack.append(']')
        elif not stack or stack.pop() != c:
            return False
    return len(stack) == 0
```
