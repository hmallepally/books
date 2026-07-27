```python
def alternating_sums(self, a: list[int]) -> list[int]:
    team1 = 0
    team2 = 0

    for i in range(len(a)):
        if i % 2 == 0:
            team1 += a[i]
        else:
            team2 += a[i]

    return [team1, team2]
# Time: O(N), Space: O(1)
```