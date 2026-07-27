```csharp
public void Backtrack(List<IList<int>> res, List<int> path, int[] nums, bool[] used)
{
    if (path.Count == nums.Length)
    {
        res.Add(new List<int>(path));
        return;
    }
    for (int i = 0; i < nums.Length; i++)
    {
        if (used[i]) continue;
        used[i] = true;
        path.Add(nums[i]);
        Backtrack(res, path, nums, used); // Recurse
        path.RemoveAt(path.Count - 1);    // Undo (backtrack)
        used[i] = false;
    }
}
```
