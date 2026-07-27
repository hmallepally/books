```python
def is_valid_sudoku(self, board: list[list[str]]) -> bool:
    seen = set()
    for i in range(9):
        for j in range(9):
            number = board[i][j]
            if number != '.':
                box_idx = (i // 3) * 3 + j // 3
                row_key = f"{number} in row {i}"
                col_key = f"{number} in col {j}"
                box_key = f"{number} in box {box_idx}"
                if row_key in seen or col_key in seen or box_key in seen:
                    return False
                seen.add(row_key)
                seen.add(col_key)
                seen.add(box_key)
    return True
```