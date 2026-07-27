```python
def add_border(self, picture: list[str]) -> list[str]:
    new_width = len(picture[0]) + 2
    result = [""] * (len(picture) + 2)

    # Build the border row
    border = '*' * new_width

    # Top border
    result[0] = border

    # Wrap each interior row with side asterisks
    for i in range(len(picture)):
        result[i + 1] = f"*{picture[i]}*"

    # Bottom border
    result[-1] = border

    return result
# Time: O(rows * cols), Space: O(rows * cols) for output
```