import os

translations = {
    "18": {
        "python": """```python
def check_subarray_sum(self, nums: list[int], k: int) -> bool:
    hash_map = {0: -1}
    total_sum = 0
    for i, num in enumerate(nums):
        total_sum += num
        mod = total_sum if k == 0 else total_sum % k
        if mod in hash_map:
            if i - hash_map[mod] > 1: return True # Length >= 2
        else:
            hash_map[mod] = i
    return False
# Time Complexity: O(N) | Space Complexity: O(min(N, K))
```""",
        "csharp": """```csharp
public bool CheckSubarraySum(int[] nums, int k) {
    Dictionary<int, int> map = new Dictionary<int, int>();
    map[0] = -1;
    int sum = 0;
    for (int i = 0; i < nums.Length; i++) {
        sum += nums[i];
        int mod = k == 0 ? sum : ((sum % k) + k) % k;
        if (map.ContainsKey(mod)) {
            if (i - map[mod] > 1) return true; // Length >= 2
        } else {
            map[mod] = i;
        }
    }
    return false;
}
// Time Complexity: O(N) | Space Complexity: O(min(N, K))
```"""
    },
    "19": {
        "python": """```python
def longest_ones(self, nums: list[int], k: int) -> int:
    left = 0
    for right in range(len(nums)):
        if nums[right] == 0: k -= 1
        if k < 0: # Over budget
            if nums[left] == 0: k += 1
            left += 1
    return len(nums) - left # Trick to return max valid length seen
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int LongestOnes(int[] nums, int k) {
    int left = 0;
    for (int right = 0; right < nums.Length; right++) {
        if (nums[right] == 0) k--;
        if (k < 0) { // Over budget
            if (nums[left++] == 0) k++;
        }
    }
    return nums.Length - left; // Trick to return max valid length seen
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    },
    "20": {
        "python": """```python
def find_duplicates(self, nums: list[int]) -> list[int]:
    res = []
    for num in nums:
        idx = abs(num) - 1
        if nums[idx] < 0: res.append(abs(num)) # Found duplicate
        else: nums[idx] = -nums[idx] # Mark seen
    return res
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public IList<int> FindDuplicates(int[] nums) {
    List<int> res = new List<int>();
    foreach (int num in nums) {
        int idx = Math.Abs(num) - 1;
        if (nums[idx] < 0) res.Add(Math.Abs(num)); // Found duplicate
        else nums[idx] = -nums[idx]; // Mark seen
    }
    return res;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    },
    "21": {
        "python": """```python
def least_interval(self, tasks: list[str], n: int) -> int:
    count = [0] * 26
    max_val = max_count = 0
    for c in tasks:
        idx = ord(c) - ord('A')
        count[idx] += 1
        if count[idx] == max_val:
            max_count += 1
        elif count[idx] > max_val:
            max_val = count[idx]
            max_count = 1
            
    empty_slots = (max_val - 1) * (n - (max_count - 1))
    available_tasks = len(tasks) - max_val * max_count
    idles = max(0, empty_slots - available_tasks)
    return len(tasks) + idles
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int LeastInterval(char[] tasks, int n) {
    int[] count = new int[26];
    int max = 0, maxCount = 0;
    foreach (char c in tasks) {
        count[c - 'A']++;
        if (count[c - 'A'] == max) maxCount++;
        else if (count[c - 'A'] > max) { max = count[c - 'A']; maxCount = 1; }
    }
    int emptySlots = (max - 1) * (n - (maxCount - 1));
    int availableTasks = tasks.Length - max * maxCount;
    int idles = Math.Max(0, emptySlots - availableTasks);
    return tasks.Length + idles;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    },
    "22": {
        "python": """```python
def insert(self, intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:
    res = []
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < new_interval[0]:
        res.append(intervals[i]) # Before
        i += 1
    while i < n and intervals[i][0] <= new_interval[1]: # Merge
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    res.append(new_interval)
    while i < n:
        res.append(intervals[i]) # After
        i += 1
    return res
# Time Complexity: O(N) | Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int[][] Insert(int[][] intervals, int[] newInterval) {
    List<int[]> res = new List<int[]>();
    int i = 0, n = intervals.Length;
    while (i < n && intervals[i][1] < newInterval[0]) res.Add(intervals[i++]); // Before
    while (i < n && intervals[i][0] <= newInterval[1]) { // Merge
        newInterval[0] = Math.Min(newInterval[0], intervals[i][0]);
        newInterval[1] = Math.Max(newInterval[1], intervals[i][1]);
        i++;
    }
    res.Add(newInterval);
    while (i < n) res.Add(intervals[i++]); // After
    return res.ToArray();
}
// Time Complexity: O(N) | Space Complexity: O(N)
```"""
    },
    "23": {
        "python": """```python
def top_k_frequent(self, nums: list[int], k: int) -> list[int]:
    from collections import Counter
    import heapq
    
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
# Time Complexity: O(N log K) | Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int[] TopKFrequent(int[] nums, int k) {
    Dictionary<int, int> count = new Dictionary<int, int>();
    foreach (int n in nums) count[n] = count.GetValueOrDefault(n, 0) + 1;
    PriorityQueue<int, int> heap = new PriorityQueue<int, int>();
    foreach (int n in count.Keys) {
        heap.Enqueue(n, count[n]);
        if (heap.Count > k) heap.Dequeue(); // Keep size K
    }
    int[] res = new int[k];
    for (int i = k - 1; i >= 0; i--) res[i] = heap.Dequeue();
    return res;
}
// Time Complexity: O(N log K) | Space Complexity: O(N)
```"""
    },
    "24": {
        "python": """```python
def first_missing_positive(self, nums: list[int]) -> int:
    i = 0
    while i < len(nums):
        # Swap to correct position if valid
        if 0 < nums[i] <= len(nums) and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        else:
            i += 1
            
    for i in range(len(nums)):
        if nums[i] != i + 1: return i + 1 # Missing
        
    return len(nums) + 1
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int FirstMissingPositive(int[] nums) {
    int i = 0;
    while (i < nums.Length) {
        // Swap to correct position if valid
        if (nums[i] > 0 && nums[i] <= nums.Length && nums[nums[i] - 1] != nums[i]) {
            int temp = nums[nums[i] - 1];
            nums[nums[i] - 1] = nums[i];
            nums[i] = temp;
        } else {
            i++;
        }
    }
    for (i = 0; i < nums.Length; i++) {
        if (nums[i] != i + 1) return i + 1; // Missing
    }
    return nums.Length + 1;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    },
    "25": {
        "python": """```python
def min_sub_array_len(self, target: int, nums: list[int]) -> int:
    left = total_sum = 0
    min_val = float('inf')
    for right in range(len(nums)):
        total_sum += nums[right]
        while total_sum >= target:
            min_val = min(min_val, right - left + 1)
            total_sum -= nums[left]
            left += 1
    return 0 if min_val == float('inf') else min_val
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int MinSubArrayLen(int target, int[] nums) {
    int left = 0, sum = 0, min = int.MaxValue;
    for (int right = 0; right < nums.Length; right++) {
        sum += nums[right];
        while (sum >= target) {
            min = Math.Min(min, right - left + 1);
            sum -= nums[left++];
        }
    }
    return min == int.MaxValue ? 0 : min;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    },
    "26": {
        "python": """```python
def find_substring(self, s: str, words: list[str]) -> list[int]:
    res = []
    if not s or not words: return res
    word_len = len(words[0])
    total_len = word_len * len(words)
    
    from collections import Counter
    counts = Counter(words)
    
    for i in range(len(s) - total_len + 1):
        seen = {}
        j = 0
        while j < len(words):
            w = s[i + j * word_len : i + (j + 1) * word_len]
            if w in counts:
                seen[w] = seen.get(w, 0) + 1
                if seen[w] > counts[w]: break
            else:
                break
            j += 1
        if j == len(words): res.append(i)
    return res
# Time Complexity: O(N * M * L) | Space Complexity: O(M)
```""",
        "csharp": """```csharp
public IList<int> FindSubstring(string s, string[] words) {
    List<int> res = new List<int>();
    if (s.Length == 0 || words.Length == 0) return res;
    int wordLen = words[0].Length, totalLen = wordLen * words.Length;
    Dictionary<string, int> counts = new Dictionary<string, int>();
    foreach (string w in words) counts[w] = counts.GetValueOrDefault(w, 0) + 1;
    
    for (int i = 0; i <= s.Length - totalLen; i++) {
        Dictionary<string, int> seen = new Dictionary<string, int>();
        int j = 0;
        while (j < words.Length) {
            string w = s.Substring(i + j * wordLen, wordLen);
            if (counts.ContainsKey(w)) {
                seen[w] = seen.GetValueOrDefault(w, 0) + 1;
                if (seen[w] > counts[w]) break;
            } else break;
            j++;
        }
        if (j == words.Length) res.Add(i);
    }
    return res;
}
// Time Complexity: O(N * M * L) | Space Complexity: O(M)
```"""
    },
    "27": {
        "python": """```python
def contains_nearby_duplicate(self, nums: list[int], k: int) -> bool:
    hash_set = set()
    for i in range(len(nums)):
        if i > k: hash_set.remove(nums[i - k - 1])
        if nums[i] in hash_set: return True
        hash_set.add(nums[i])
    return False
# Time Complexity: O(N) | Space Complexity: O(K)
```""",
        "csharp": """```csharp
public bool ContainsNearbyDuplicate(int[] nums, int k) {
    HashSet<int> set = new HashSet<int>();
    for (int i = 0; i < nums.Length; i++) {
        if (i > k) set.Remove(nums[i - k - 1]);
        if (!set.Add(nums[i])) return true;
    }
    return false;
}
// Time Complexity: O(N) | Space Complexity: O(K)
```"""
    },
    "28": {
        "python": """```python
def number_of_subarrays(self, nums: list[int], k: int) -> int:
    from collections import defaultdict
    hash_map = defaultdict(int)
    hash_map[0] = 1
    total_sum = count = 0
    for num in nums:
        total_sum += num % 2
        count += hash_map[total_sum - k]
        hash_map[total_sum] += 1
    return count
# Time Complexity: O(N) | Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int NumberOfSubarrays(int[] nums, int k) {
    Dictionary<int, int> map = new Dictionary<int, int>();
    map[0] = 1;
    int sum = 0, count = 0;
    foreach (int num in nums) {
        sum += num % 2;
        count += map.GetValueOrDefault(sum - k, 0);
        map[sum] = map.GetValueOrDefault(sum, 0) + 1;
    }
    return count;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```"""
    },
    "29": {
        "python": """```python
def max_frequency(self, nums: list[int], k: int) -> int:
    nums.sort()
    left = total_sum = 0
    for right in range(len(nums)):
        total_sum += nums[right]
        if nums[right] * (right - left + 1) - total_sum > k:
            total_sum -= nums[left]
            left += 1
    return len(nums) - left
# Time Complexity: O(N log N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int MaxFrequency(int[] nums, int k) {
    Array.Sort(nums);
    int left = 0;
    long sum = 0;
    for (int right = 0; right < nums.Length; right++) {
        sum += nums[right];
        if ((long)nums[right] * (right - left + 1) - sum > k) {
            sum -= nums[left++];
        }
    }
    return nums.Length - left;
}
// Time Complexity: O(N log N) | Space Complexity: O(1)
```"""
    },
    "30": {
        "python": """```python
def subarrays_with_k_distinct(self, nums: list[int], k: int) -> int:
    return self._at_most_k(nums, k) - self._at_most_k(nums, k - 1)

def _at_most_k(self, nums: list[int], k: int) -> int:
    count = [0] * (len(nums) + 1)
    left = res = distinct = 0
    for right in range(len(nums)):
        if count[nums[right]] == 0: distinct += 1
        count[nums[right]] += 1
        while distinct > k:
            count[nums[left]] -= 1
            if count[nums[left]] == 0: distinct -= 1
            left += 1
        res += right - left + 1
    return res
# Time Complexity: O(N) | Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int SubarraysWithKDistinct(int[] nums, int k) {
    return AtMostK(nums, k) - AtMostK(nums, k - 1);
}
private int AtMostK(int[] nums, int k) {
    int[] count = new int[nums.Length + 1];
    int left = 0, res = 0, distinct = 0;
    for (int right = 0; right < nums.Length; right++) {
        if (count[nums[right]]++ == 0) distinct++;
        while (distinct > k) {
            if (--count[nums[left++]] == 0) distinct--;
        }
        res += right - left + 1;
    }
    return res;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```"""
    },
    "31": {
        "python": """```python
def longest_palindrome(self, s: str) -> str:
    start = end = 0
    for i in range(len(s)):
        len1 = self._expand(s, i, i)
        len2 = self._expand(s, i, i + 1)
        length = max(len1, len2)
        if length > end - start:
            start = i - (length - 1) // 2
            end = i + length // 2
    return s[start:end + 1]

def _expand(self, s: str, l: int, r: int) -> int:
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1; r += 1
    return r - l - 1
# Time Complexity: O(N^2) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public string LongestPalindrome(string s) {
    int start = 0, end = 0;
    for (int i = 0; i < s.Length; i++) {
        int len1 = Expand(s, i, i);
        int len2 = Expand(s, i, i + 1);
        int len = Math.Max(len1, len2);
        if (len > end - start) {
            start = i - (len - 1) / 2;
            end = i + len / 2;
        }
    }
    return s.Substring(start, end - start + 1);
}
private int Expand(string s, int L, int R) {
    while (L >= 0 && R < s.Length && s[L] == s[R]) { L--; R++; }
    return R - L - 1;
}
// Time Complexity: O(N^2) | Space Complexity: O(1)
```"""
    },
    "32": {
        "python": """```python
def three_sum(self, nums: list[int]) -> list[list[int]]:
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            total = nums[i] + nums[l] + nums[r]
            if total == 0:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l+1]: l += 1
                while l < r and nums[r] == nums[r-1]: r -= 1
                l += 1; r -= 1
            elif total < 0: l += 1
            else: r -= 1
    return res
# Time Complexity: O(N^2) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public IList<IList<int>> ThreeSum(int[] nums) {
    Array.Sort(nums);
    IList<IList<int>> res = new List<IList<int>>();
    for (int i = 0; i < nums.Length - 2; i++) {
        if (i > 0 && nums[i] == nums[i-1]) continue;
        int L = i + 1, R = nums.Length - 1;
        while (L < R) {
            int sum = nums[i] + nums[L] + nums[R];
            if (sum == 0) {
                res.Add(new List<int>{nums[i], nums[L], nums[R]});
                while (L < R && nums[L] == nums[L+1]) L++;
                while (L < R && nums[R] == nums[R-1]) R--;
                L++; R--;
            }
            else if (sum < 0) L++;
            else R--;
        }
    }
    return res;
}
// Time Complexity: O(N^2) | Space Complexity: O(1)
```"""
    },
    "33": {
        "python": """```python
def four_sum(self, nums: list[int], target: int) -> list[list[int]]:
    nums.sort()
    res = []
    for i in range(len(nums) - 3):
        if i > 0 and nums[i] == nums[i-1]: continue
        for j in range(i + 1, len(nums) - 2):
            if j > i + 1 and nums[j] == nums[j-1]: continue
            l, r = j + 1, len(nums) - 1
            while l < r:
                total = nums[i] + nums[j] + nums[l] + nums[r]
                if total == target:
                    res.append([nums[i], nums[j], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]: l += 1
                    while l < r and nums[r] == nums[r-1]: r -= 1
                    l += 1; r -= 1
                elif total < target: l += 1
                else: r -= 1
    return res
# Time Complexity: O(N^3) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public IList<IList<int>> FourSum(int[] nums, int target) {
    Array.Sort(nums);
    IList<IList<int>> res = new List<IList<int>>();
    for (int i = 0; i < nums.Length - 3; i++) {
        if (i > 0 && nums[i] == nums[i-1]) continue;
        for (int j = i + 1; j < nums.Length - 2; j++) {
            if (j > i + 1 && nums[j] == nums[j-1]) continue;
            int L = j + 1, R = nums.Length - 1;
            while (L < R) {
                long sum = (long)nums[i] + nums[j] + nums[L] + nums[R];
                if (sum == target) {
                    res.Add(new List<int>{nums[i], nums[j], nums[L], nums[R]});
                    while (L < R && nums[L] == nums[L+1]) L++;
                    while (L < R && nums[R] == nums[R-1]) R--;
                    L++; R--;
                }
                else if (sum < target) L++;
                else R--;
            }
        }
    }
    return res;
}
// Time Complexity: O(N^3) | Space Complexity: O(1)
```"""
    },
    "34": {
        "python": """```python
def num_distinct_islands(self, grid: list[list[int]]) -> int:
    hash_set = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                path = []
                self._dfs(grid, i, j, "S", path) # Start with 'S'
                hash_set.add("".join(path))
    return len(hash_set)

def _dfs(self, grid: list[list[int]], r: int, c: int, dir_str: str, path: list[str]) -> None:
    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] == 0: return
    grid[r][c] = 0 # mark visited
    path.append(dir_str)
    self._dfs(grid, r + 1, c, "D", path)
    self._dfs(grid, r - 1, c, "U", path)
    self._dfs(grid, r, c + 1, "R", path)
    self._dfs(grid, r, c - 1, "L", path)
    path.append("B") # Backtrack to distinguish paths
# Time Complexity: O(R * C) | Space Complexity: O(R * C)
```""",
        "csharp": """```csharp
public int NumDistinctIslands(int[][] grid) {
    HashSet<string> set = new HashSet<string>();
    for (int i = 0; i < grid.Length; i++) {
        for (int j = 0; j < grid[0].Length; j++) {
            if (grid[i][j] == 1) {
                StringBuilder sb = new StringBuilder();
                Dfs(grid, i, j, "S", sb); // Start with 'S'
                set.Add(sb.ToString());
            }
        }
    }
    return set.Count;
}
private void Dfs(int[][] grid, int r, int c, string dir, StringBuilder sb) {
    if (r < 0 || c < 0 || r >= grid.Length || c >= grid[0].Length || grid[r][c] == 0) return;
    grid[r][c] = 0; // mark visited
    sb.Append(dir);
    Dfs(grid, r + 1, c, "D", sb);
    Dfs(grid, r - 1, c, "U", sb);
    Dfs(grid, r, c + 1, "R", sb);
    Dfs(grid, r, c - 1, "L", sb);
    sb.Append("B"); // Backtrack to distinguish paths
}
// Time Complexity: O(R * C) | Space Complexity: O(R * C)
```"""
    }
}

base_dir = r"C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\12-hashmaps-sliding-windows\snippets"
for k, v in translations.items():
    with open(os.path.join(base_dir, "python", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["python"])
    with open(os.path.join(base_dir, "csharp", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["csharp"])

print("Translations for chapter 12 fully applied!")
