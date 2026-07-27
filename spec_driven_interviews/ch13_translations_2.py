import os

translations = {
    "18": {
        "python": """```python
def coin_change(self, coins: list[int], amount: int) -> int:
    dp = [amount + 1] * (amount + 1) # Fill with max invalid value
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
                
    return -1 if dp[amount] > amount else dp[amount]
# Time Complexity: O(Amount * N)
# Space Complexity: O(Amount)
```""",
        "csharp": """```csharp
public int CoinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Array.Fill(dp, amount + 1); // Fill with max invalid value
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        foreach (int coin in coins) {
            if (i >= coin) {
                dp[i] = Math.Min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}
// Time Complexity: O(Amount * N)
// Space Complexity: O(Amount)
```"""
    },
    "19": {
        "python": """```python
def rob(self, nums: list[int]) -> int:
    if not nums: return 0
    prev1 = 0 # max so far excluding current
    prev2 = 0 # max so far including current (-2)
    
    for num in nums:
        temp = max(prev1, prev2 + num) # rob or don't rob
        prev2 = prev1
        prev1 = temp
        
    return prev1
# Time Complexity: O(N)
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int Rob(int[] nums) {
    if (nums == null || nums.Length == 0) return 0;
    int prev1 = 0; // max so far excluding current
    int prev2 = 0; // max so far including current (-2)
    
    foreach (int num in nums) {
        int temp = Math.Max(prev1, prev2 + num); // rob or don't rob
        prev2 = prev1;
        prev1 = temp;
    }
    return prev1;
}
// Time Complexity: O(N)
// Space Complexity: O(1)
```"""
    },
    "20": {
        "python": """```python
def is_match(self, s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Match empty string with patterns like a*b*
    for j in range(1, n + 1):
        if p[j - 1] == '*': dp[0][j] = dp[0][j - 2]
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1] # Single char match
            elif p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2] # Match zero times
                # If preceding char matches, match one or more times
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
                    
    return dp[m][n]
# Time Complexity: O(M * N)
# Space Complexity: O(M * N)
```""",
        "csharp": """```csharp
public bool IsMatch(string s, string p) {
    int m = s.Length, n = p.Length;
    bool[][] dp = new bool[m + 1][];
    for(int i=0; i<=m; i++) dp[i] = new bool[n + 1];
    dp[0][0] = true;
    
    // Match empty string with patterns like a*b*
    for (int j = 1; j <= n; j++) {
        if (p[j - 1] == '*') dp[0][j] = dp[0][j - 2];
    }
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (p[j - 1] == '.' || p[j - 1] == s[i - 1]) {
                dp[i][j] = dp[i - 1][j - 1]; // Single char match
            } else if (p[j - 1] == '*') {
                dp[i][j] = dp[i][j - 2]; // Match zero times
                // If preceding char matches, match one or more times
                if (p[j - 2] == '.' || p[j - 2] == s[i - 1]) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j];
                }
            }
        }
    }
    return dp[m][n];
}
// Time Complexity: O(M * N)
// Space Complexity: O(M * N)
```"""
    },
    "21": {
        "python": """```python
def find_order(self, num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    in_degree = [0] * num_courses
    adj = [[] for _ in range(num_courses)]
    
    for dest, src in prerequisites:
        adj[src].append(dest)
        in_degree[dest] += 1
        
    from collections import deque
    q = deque(i for i in range(num_courses) if in_degree[i] == 0)
    
    res = []
    while q:
        curr = q.popleft()
        res.append(curr)
        for nxt in adj[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                q.append(nxt)
                
    return res if len(res) == num_courses else [] # If not all courses taken, cycle exists
# Time Complexity: O(V + E)
# Space Complexity: O(V + E)
```""",
        "csharp": """```csharp
public int[] FindOrder(int numCourses, int[][] prerequisites) {
    var inDegree = new int[numCourses];
    var adj = new List<List<int>>();
    for (int i = 0; i < numCourses; i++) adj.Add(new List<int>());
    
    foreach (int[] p in prerequisites) {
        adj[p[1]].Add(p[0]);
        inDegree[p[0]]++;
    }
    
    Queue<int> q = new Queue<int>();
    for (int i = 0; i < numCourses; i++) {
        if (inDegree[i] == 0) q.Enqueue(i);
    }
    
    int[] res = new int[numCourses];
    int idx = 0;
    while (q.Count > 0) {
        int curr = q.Dequeue();
        res[idx++] = curr;
        foreach (int next in adj[curr]) {
            if (--inDegree[next] == 0) q.Enqueue(next);
        }
    }
    return idx == numCourses ? res : new int[0]; // If not all courses taken, cycle exists
}
// Time Complexity: O(V + E)
// Space Complexity: O(V + E)
```"""
    },
    "22": {
        "python": """```python
def can_partition(self, nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2 != 0: return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        # Iterate backwards to avoid reusing the same element
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
            
    return dp[target]
# Time Complexity: O(N * Target)
# Space Complexity: O(Target)
```""",
        "csharp": """```csharp
public bool CanPartition(int[] nums) {
    int sum = 0;
    foreach (int num in nums) sum += num;
    if (sum % 2 != 0) return false;
    
    int target = sum / 2;
    bool[] dp = new bool[target + 1];
    dp[0] = true;
    
    foreach (int num in nums) {
        // Iterate backwards to avoid reusing the same element
        for (int j = target; j >= num; j--) {
            dp[j] = dp[j] || dp[j - num];
        }
    }
    return dp[target];
}
// Time Complexity: O(N * Target)
// Space Complexity: O(Target)
```"""
    },
    "23": {
        "python": """```python
def num_decodings(self, s: str) -> int:
    if not s or s[0] == '0': return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    
    for i in range(2, n + 1):
        one_digit = int(s[i - 1:i])
        two_digits = int(s[i - 2:i])
        
        if 1 <= one_digit <= 9:
            dp[i] += dp[i - 1]
        if 10 <= two_digits <= 26:
            dp[i] += dp[i - 2]
            
    return dp[n]
# Time Complexity: O(N)
# Space Complexity: O(N) which can be optimized to O(1)
```""",
        "csharp": """```csharp
public int NumDecodings(string s) {
    if (string.IsNullOrEmpty(s) || s[0] == '0') return 0;
    int n = s.Length;
    int[] dp = new int[n + 1];
    dp[0] = 1; 
    dp[1] = 1;
    
    for (int i = 2; i <= n; i++) {
        int oneDigit = int.Parse(s.Substring(i - 1, 1));
        int twoDigits = int.Parse(s.Substring(i - 2, 2));
        
        if (oneDigit >= 1 && oneDigit <= 9) {
            dp[i] += dp[i - 1];
        }
        if (twoDigits >= 10 && twoDigits <= 26) {
            dp[i] += dp[i - 2];
        }
    }
    return dp[n];
}
// Time Complexity: O(N)
// Space Complexity: O(N) which can be optimized to O(1)
```"""
    },
    "24": {
        "python": """```python
class StockSpanner:
    def __init__(self):
        # Array holds [price, span]
        self.stack = []
        
    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1] # Accumulate previous spans
        self.stack.append([price, span])
        return span
# Time Complexity: Amortized O(1) per next() call
# Space Complexity: O(N)
```""",
        "csharp": """```csharp
public class StockSpanner {
    // Stack holds {price, span}
    private Stack<int[]> stack = new Stack<int[]>(); 
    
    public int Next(int price) {
        int span = 1;
        while (stack.Count > 0 && stack.Peek()[0] <= price) {
            span += stack.Pop()[1]; // Accumulate previous spans
        }
        stack.Push(new int[]{price, span});
        return span;
    }
}
// Time Complexity: Amortized O(1) per next() call
// Space Complexity: O(N)
```"""
    },
    "25": {
        "python": """```python
def length_of_lis(self, nums: list[int]) -> int:
    tails = [0] * len(nums)
    size = 0
    for x in nums:
        left, right = 0, size
        while left != right:
            mid = left + (right - left) // 2
            if tails[mid] < x:
                left = mid + 1
            else:
                right = mid
        tails[left] = x
        if left == size: size += 1 # Found a larger element, expand LIS
    return size
# Time Complexity: O(N log N)
# Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int LengthOfLIS(int[] nums) {
    int[] tails = new int[nums.Length];
    int size = 0;
    foreach (int x in nums) {
        int left = 0, right = size;
        while (left != right) {
            int mid = left + (right - left) / 2;
            if (tails[mid] < x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        tails[left] = x;
        if (left == size) size++; // Found a larger element, expand LIS
    }
    return size;
}
// Time Complexity: O(N log N)
// Space Complexity: O(N)
```"""
    },
    "26": {
        "python": """```python
def find_min(self, nums: list[int]) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]: left = mid + 1
        else: right = mid
    return nums[left]
# Time Complexity: O(log N)
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int FindMin(int[] nums) {
    int left = 0, right = nums.Length - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[right]) left = mid + 1;
        else right = mid;
    }
    return nums[left];
}
// Time Complexity: O(log N)
// Space Complexity: O(1)
```"""
    },
    "27": {
        "python": """```python
def kth_smallest(self, matrix: list[list[int]], k: int) -> int:
    n = len(matrix)
    left, right = matrix[0][0], matrix[n-1][n-1]
    while left < right:
        mid = left + (right - left) // 2
        count = self._count_less_equal(matrix, mid)
        if count < k: left = mid + 1
        else: right = mid
    return left

def _count_less_equal(self, matrix: list[list[int]], target: int) -> int:
    n, i, j, count = len(matrix), len(matrix) - 1, 0, 0
    while i >= 0 and j < n:
        if matrix[i][j] <= target:
            count += i + 1
            j += 1
        else:
            i -= 1
    return count
# Time Complexity: O(N log(Max - Min))
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int KthSmallest(int[][] matrix, int k) {
    int n = matrix.Length;
    int left = matrix[0][0], right = matrix[n-1][n-1];
    while (left < right) {
        int mid = left + (right - left) / 2;
        int count = CountLessEqual(matrix, mid);
        if (count < k) left = mid + 1;
        else right = mid;
    }
    return left;
}
private int CountLessEqual(int[][] matrix, int target) {
    int n = matrix.Length, i = n - 1, j = 0, count = 0;
    while (i >= 0 && j < n) {
        if (matrix[i][j] <= target) { count += i + 1; j++; }
        else { i--; }
    }
    return count;
}
// Time Complexity: O(N log(Max - Min))
// Space Complexity: O(1)
```"""
    },
    "28": {
        "python": """```python
def jump(self, nums: list[int]) -> int:
    jumps = current_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps
# Time Complexity: O(N)
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int Jump(int[] nums) {
    int jumps = 0, currentEnd = 0, farthest = 0;
    for (int i = 0; i < nums.Length - 1; i++) {
        farthest = Math.Max(farthest, i + nums[i]);
        if (i == currentEnd) {
            jumps++;
            currentEnd = farthest;
        }
    }
    return jumps;
}
// Time Complexity: O(N)
// Space Complexity: O(1)
```"""
    },
    "29": {
        "python": """```python
def unique_paths(self, m: int, n: int) -> int:
    dp = [[0] * n for _ in range(m)]
    for i in range(m): dp[i][0] = 1
    for j in range(n): dp[0][j] = 1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]
# Time Complexity: O(M * N)
# Space Complexity: O(M * N) (can be optimized to O(N))
```""",
        "csharp": """```csharp
public int UniquePaths(int m, int n) {
    int[][] dp = new int[m][];
    for (int i = 0; i < m; i++) {
        dp[i] = new int[n];
        dp[i][0] = 1;
    }
    for (int j = 0; j < n; j++) dp[0][j] = 1;
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
}
// Time Complexity: O(M * N)
// Space Complexity: O(M * N) (can be optimized to O(N))
```"""
    },
    "30": {
        "python": """```python
def max_sub_array(self, nums: list[int]) -> int:
    max_sum = current_sum = nums[0]
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    return max_sum
# Time Complexity: O(N)
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int MaxSubArray(int[] nums) {
    int maxSum = nums[0], currentSum = nums[0];
    for (int i = 1; i < nums.Length; i++) {
        currentSum = Math.Max(nums[i], currentSum + nums[i]);
        maxSum = Math.Max(maxSum, currentSum);
    }
    return maxSum;
}
// Time Complexity: O(N)
// Space Complexity: O(1)
```"""
    },
    "31": {
        "python": """```python
def climb_stairs(self, n: int) -> int:
    if n <= 2: return n
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1
# Time Complexity: O(N)
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int ClimbStairs(int n) {
    if (n <= 2) return n;
    int prev2 = 1, prev1 = 2;
    for (int i = 3; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
// Time Complexity: O(N)
// Space Complexity: O(1)
```"""
    },
    "32": {
        "python": """```python
def largest_rectangle_area(self, heights: list[int]) -> int:
    stack = []
    max_area = 0
    n = len(heights)
    for i in range(n + 1):
        h = 0 if i == n else heights[i]
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area
# Time Complexity: O(N)
# Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int LargestRectangleArea(int[] heights) {
    Stack<int> stack = new Stack<int>();
    int maxArea = 0, n = heights.Length;
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];
        while (stack.Count > 0 && h < heights[stack.Peek()]) {
            int height = heights[stack.Pop()];
            int width = stack.Count == 0 ? i : i - stack.Peek() - 1;
            maxArea = Math.Max(maxArea, height * width);
        }
        stack.Push(i);
    }
    return maxArea;
}
// Time Complexity: O(N)
// Space Complexity: O(N)
```"""
    },
    "33": {
        "python": """```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def merge_k_lists(self, lists: list[ListNode]) -> ListNode:
    import heapq
    
    # Python heapq requires a way to break ties if vals are equal.
    # We can use id(node) or an index.
    pq = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(pq, (head.val, i, head))
            
    dummy = ListNode(0)
    curr = dummy
    
    while pq:
        val, i, min_node = heapq.heappop(pq)
        curr.next = min_node
        curr = curr.next
        if min_node.next:
            heapq.heappush(pq, (min_node.next.val, i, min_node.next))
            
    return dummy.next
# Time Complexity: O(N log K)
# Space Complexity: O(K)
```""",
        "csharp": """```csharp
public ListNode MergeKLists(ListNode[] lists) {
    PriorityQueue<ListNode, int> pq = new PriorityQueue<ListNode, int>();
    foreach (ListNode head in lists) {
        if (head != null) pq.Enqueue(head, head.val);
    }
    ListNode dummy = new ListNode(0), curr = dummy;
    while (pq.Count > 0) {
        ListNode minNode = pq.Dequeue();
        curr.next = minNode;
        curr = curr.next;
        if (minNode.next != null) pq.Enqueue(minNode.next, minNode.next.val);
    }
    return dummy.next;
}
// Time Complexity: O(N log K)
// Space Complexity: O(K)
```"""
    },
    "34": {
        "python": """```python
def longest_valid_parentheses(self, s: str) -> int:
    max_len = 0
    dp = [0] * len(s)
    for i in range(1, len(s)):
        if s[i] == ')':
            if s[i - 1] == '(':
                dp[i] = (dp[i - 2] if i >= 2 else 0) + 2
            elif i - dp[i - 1] > 0 and s[i - dp[i - 1] - 1] == '(':
                dp[i] = dp[i - 1] + (dp[i - dp[i - 1] - 2] if (i - dp[i - 1]) >= 2 else 0) + 2
            max_len = max(max_len, dp[i])
    return max_len
# Time Complexity: O(N)
# Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int LongestValidParentheses(string s) {
    int maxLen = 0;
    int[] dp = new int[s.Length];
    for (int i = 1; i < s.Length; i++) {
        if (s[i] == ')') {
            if (s[i - 1] == '(') {
                dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
            } else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(') {
                dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
            }
            maxLen = Math.Max(maxLen, dp[i]);
        }
    }
    return maxLen;
}
// Time Complexity: O(N)
// Space Complexity: O(N)
```"""
    },
    "35": {
        "python": """```python
def max_area(self, height: list[int]) -> int:
    max_area = 0
    left, right = 0, len(height) - 1
    while left < right:
        w = right - left
        h = min(height[left], height[right])
        max_area = max(max_area, w * h)
        if height[left] < height[right]: left += 1
        else: right -= 1
    return max_area
# Time Complexity: O(N)
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int MaxArea(int[] height) {
    int maxArea = 0;
    int left = 0, right = height.Length - 1;
    while (left < right) {
        int w = right - left;
        int h = Math.Min(height[left], height[right]);
        maxArea = Math.Max(maxArea, w * h);
        if (height[left] < height[right]) left++;
        else right--;
    }
    return maxArea;
}
// Time Complexity: O(N)
// Space Complexity: O(1)
```"""
    }
}

base_dir = r"C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\13-optimization-dp\snippets"
for k, v in translations.items():
    with open(os.path.join(base_dir, "python", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["python"])
    with open(os.path.join(base_dir, "csharp", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["csharp"])

print("Translations for chapter 13 fully applied!")
