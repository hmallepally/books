```java
public void backtrack(List<List<Integer>> res, List<Integer> path, int[] nums, boolean[] used) {
    if (path.size() == nums.length) {
        res.add(new ArrayList<>(path));
        return;
    }
    for (int i = 0; i < nums.length; i++) {
        if (used[i]) continue;
        used[i] = true;
        path.add(nums[i]);
        backtrack(res, path, nums, used); // Recurse
        path.remove(path.size() - 1);     // Undo (backtrack)
        used[i] = false;
    }
}
```
