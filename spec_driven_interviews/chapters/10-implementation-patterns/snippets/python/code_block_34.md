```python
def reverse_in_parentheses(self, s: str) -> str:
    stack = [[]]

    for c in s:
        if c == '(':
            stack.append([]) # Start new nested context
        elif c == ')':
            inner = stack.pop()  # Pop innermost context
            inner.reverse()       # Reverse it
            stack[-1].extend(inner) # Append to enclosing context
        else:
            stack[-1].append(c)  # Accumulate character

    return "".join(stack[0])
# Time: O(N^2) worst case for nested reversals, Space: O(N)
```