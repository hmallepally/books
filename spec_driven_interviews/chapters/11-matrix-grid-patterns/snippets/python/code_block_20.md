```python
def flip_and_invert_image(self, image: list[list[int]]) -> list[list[int]]:
    for row in image:
        left, right = 0, len(row) - 1
        while left <= right:
            row[left], row[right] = row[right] ^ 1, row[left] ^ 1
            left += 1; right -= 1
    return image
```