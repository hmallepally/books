import os

translations = {
    "1": {
        "python": """```python
# Standard Binary Search
def binary_search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target: return mid
        elif nums[mid] < target: left = mid + 1
        else: right = mid - 1
    return -1

# Binary Search on Answer Space (Leftmost valid)
def binary_search_answer_space(min_val: int, max_val: int) -> int:
    left, right = min_val, max_val
    best = -1
    while left <= right:
        mid = left + (right - left) // 2
        if is_valid(mid):
            best = mid
            right = mid - 1 # Try to find a smaller valid answer
        else:
            left = mid + 1
    return best
```""",
        "csharp": """```csharp
// Standard Binary Search
int BinarySearch(int[] nums, int target) {
    int left = 0, right = nums.Length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// Binary Search on Answer Space (Leftmost valid)
int BinarySearchAnswerSpace(int min, int max) {
    int left = min, right = max;
    int best = -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (IsValid(mid)) {
            best = mid;
            right = mid - 1; // Try to find a smaller valid answer
        } else {
            left = mid + 1;
        }
    }
    return best;
}
```"""
    },
    "2": {
        "python": """```python
def next_greater_element(self, nums: list[int]) -> list[int]:
    n = len(nums)
    result = [-1] * n
    stack = [] # stores indices
    for i in range(n):
        # Maintain strictly decreasing stack
        while stack and nums[i] > nums[stack[-1]]:
            prev_index = stack.pop()
            result[prev_index] = nums[i] # Found next greater!
        stack.append(i)
    return result
```""",
        "csharp": """```csharp
public int[] NextGreaterElement(int[] nums) {
    int n = nums.Length;
    int[] result = new int[n];
    Array.Fill(result, -1);
    Stack<int> stack = new Stack<int>(); // stores indices
    for (int i = 0; i < n; i++) {
        // Maintain strictly decreasing stack
        while (stack.Count > 0 && nums[i] > nums[stack.Peek()]) {
            int prevIndex = stack.Pop();
            result[prevIndex] = nums[i]; // Found next greater!
        }
        stack.Push(i);
    }
    return result;
}
```"""
    },
    "3": {
        "python": """```python
def dp_state_compression(self, nums: list[int]) -> int:
    if not nums: return 0
    prev2 = 0 # dp[i-2]
    prev1 = nums[0] # dp[i-1]
    for i in range(1, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = curr
    return prev1
```""",
        "csharp": """```csharp
public int DpStateCompression(int[] nums) {
    if (nums.Length == 0) return 0;
    int prev2 = 0; // dp[i-2]
    int prev1 = nums[0]; // dp[i-1]
    for (int i = 1; i < nums.Length; i++) {
        int curr = Math.Max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```"""
    },
    "4": {
        "python": """```python
def bfs_level(self, start: 'Node', target: 'Node') -> int:
    from collections import deque
    queue = deque([start])
    visited = {start}
    
    level = 0
    while queue:
        size = len(queue)
        for _ in range(size):
            curr = queue.popleft()
            if curr == target: return level
            
            for neighbor in curr.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        level += 1 # Increment level after exploring all nodes at current depth
    return -1
```""",
        "csharp": """```csharp
public int BfsLevel(Node start, Node target) {
    Queue<Node> queue = new Queue<Node>();
    HashSet<Node> visited = new HashSet<Node>();
    queue.Enqueue(start);
    visited.Add(start);
    
    int level = 0;
    while (queue.Count > 0) {
        int size = queue.Count;
        for (int i = 0; i < size; i++) {
            Node curr = queue.Dequeue();
            if (curr.Equals(target)) return level;
            
            foreach (Node neighbor in curr.neighbors) {
                if (!visited.Contains(neighbor)) {
                    visited.Add(neighbor);
                    queue.Enqueue(neighbor);
                }
            }
        }
        level++; // Increment level after exploring all nodes at current depth
    }
    return -1;
}
```"""
    },
    "5": {
        "python": """```python
def topological_sort(self, num_nodes: int, edges: list[list[int]]) -> list[int]:
    from collections import deque
    adj = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    
    for u, v in edges:
        adj[v].append(u) # v -> u
        in_degree[u] += 1
        
    queue = deque(i for i in range(num_nodes) if in_degree[i] == 0)
    
    order = []
    while queue:
        curr = queue.popleft()
        order.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    return order if len(order) == num_nodes else [] # Empty if cycle exists
```""",
        "csharp": """```csharp
public IList<int> TopologicalSort(int numNodes, int[][] edges) {
    var adj = new List<List<int>>();
    int[] inDegree = new int[numNodes];
    for (int i = 0; i < numNodes; i++) adj.Add(new List<int>());
    
    foreach (int[] edge in edges) {
        adj[edge[1]].Add(edge[0]); // edge[1] -> edge[0]
        inDegree[edge[0]]++;
    }
    
    var queue = new Queue<int>();
    for (int i = 0; i < numNodes; i++) {
        if (inDegree[i] == 0) queue.Enqueue(i);
    }
    
    List<int> order = new List<int>();
    while (queue.Count > 0) {
        int curr = queue.Dequeue();
        order.Add(curr);
        foreach (int neighbor in adj[curr]) {
            if (--inDegree[neighbor] == 0) {
                queue.Enqueue(neighbor);
            }
        }
    }
    return order.Count == numNodes ? order : new List<int>(); // Empty if cycle exists
}
```"""
    },
    "6": {
        "python": """```python
def search(self, nums: list[int], target: int) -> int:
    if not nums: return -1
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target: return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1 # Target is in the sorted left half
            else:
                left = mid + 1 # Target must be in the right half
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1 # Target is in the sorted right half
            else:
                right = mid - 1 # Target must be in the left half
    return -1
# Time Complexity: O(log N)
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int Search(int[] nums, int target) {
    if (nums == null || nums.Length == 0) return -1;
    int left = 0, right = nums.Length - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        
        // Left half is sorted
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1; // Target is in the sorted left half
            } else {
                left = mid + 1; // Target must be in the right half
            }
        } 
        // Right half is sorted
        else {
            if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1; // Target is in the sorted right half
            } else {
                right = mid - 1; // Target must be in the left half
            }
        }
    }
    return -1;
}
// Time Complexity: O(log N)
// Space Complexity: O(1)
```"""
    },
    "7": {
        "python": """```python
def max_sliding_window(self, nums: list[int], k: int) -> list[int]:
    if not nums or k <= 0: return []
    n = len(nums)
    res = [0] * (n - k + 1)
    res_index = 0
    from collections import deque
    q = deque()
    
    for i in range(n):
        # Remove indices outside the current window
        if q and q[0] < i - k + 1:
            q.popleft()
        # Remove smaller elements (maintain decreasing order)
        while q and nums[q[-1]] < nums[i]:
            q.pop()
        q.append(i)
        
        # Record max for the window
        if i >= k - 1:
            res[res_index] = nums[q[0]]
            res_index += 1
            
    return res
# Time Complexity: O(N) since each element is pushed/popped at most once
# Space Complexity: O(K) for the deque
```""",
        "csharp": """```csharp
public int[] MaxSlidingWindow(int[] nums, int k) {
    if (nums == null || k <= 0) return new int[0];
    int n = nums.Length;
    int[] res = new int[n - k + 1];
    int resIndex = 0;
    LinkedList<int> q = new LinkedList<int>();
    
    for (int i = 0; i < n; i++) {
        // Remove indices outside the current window
        if (q.Count > 0 && q.First.Value < i - k + 1) {
            q.RemoveFirst();
        }
        // Remove smaller elements (maintain decreasing order)
        while (q.Count > 0 && nums[q.Last.Value] < nums[i]) {
            q.RemoveLast();
        }
        q.AddLast(i);
        
        // Record max for the window
        if (i >= k - 1) {
            res[resIndex++] = nums[q.First.Value];
        }
    }
    return res;
}
// Time Complexity: O(N) since each element is pushed/popped at most once
// Space Complexity: O(K) for the deque
```"""
    },
    "8": {
        "python": """```python
def longest_common_subsequence(self, text1: str, text2: str) -> int:
    if len(text1) < len(text2): return self.longest_common_subsequence(text2, text1)
    m, n = len(text1), len(text2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
        curr = [0] * (n + 1)
        
    return prev[n]
# Time Complexity: O(M * N)
# Space Complexity: O(min(M, N)) - Space compressed DP as taught in the vocabulary section.
```""",
        "csharp": """```csharp
public int LongestCommonSubsequence(string text1, string text2) {
    if (text1.Length < text2.Length) return LongestCommonSubsequence(text2, text1);
    int m = text1.Length, n = text2.Length;
    var prev = new int[n + 1];
    var curr = new int[n + 1];
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            curr[j] = text1[i - 1] == text2[j - 1]
                ? prev[j - 1] + 1
                : Math.Max(prev[j], curr[j - 1]);
        }
        var temp = prev; prev = curr; curr = temp;
        Array.Fill(curr, 0);
    }
    return prev[n];
}
// Time Complexity: O(M * N)
// Space Complexity: O(min(M, N)) - Space compressed DP as taught in the vocabulary section.
```"""
    },
    "9": {
        "python": """```python
def max_coins(self, nums: list[int]) -> int:
    n = len(nums)
    arr = [1] + nums + [1] # Padding with 1s
    
    dp = [[0] * (n + 2) for _ in range(n + 2)]
    
    # len_ is the length of the interval strictly between i and j
    for len_ in range(1, n + 1):
        for i in range(n - len_ + 1):
            j = i + len_ + 1
            # k is the index of the LAST balloon to burst in (i, j)
            for k in range(i + 1, j):
                coins = arr[i] * arr[k] * arr[j] + dp[i][k] + dp[k][j]
                dp[i][j] = max(dp[i][j], coins)
                
    return dp[0][n + 1]
# Time Complexity: O(N^3)
# Space Complexity: O(N^2)
```""",
        "csharp": """```csharp
public int MaxCoins(int[] nums) {
    int n = nums.Length;
    int[] arr = new int[n + 2];
    arr[0] = 1; arr[n + 1] = 1; // Padding with 1s
    for (int i = 0; i < n; i++) arr[i + 1] = nums[i];
    
    int[][] dp = new int[n + 2][];
    for(int i=0; i<n+2; i++) dp[i] = new int[n+2];
    
    // len is the length of the interval strictly between i and j
    for (int len = 1; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            int j = i + len + 1;
            // k is the index of the LAST balloon to burst in (i, j)
            for (int k = i + 1; k < j; k++) {
                int coins = arr[i] * arr[k] * arr[j] + dp[i][k] + dp[k][j];
                dp[i][j] = Math.Max(dp[i][j], coins);
            }
        }
    }
    return dp[0][n + 1];
}
// Time Complexity: O(N^3)
// Space Complexity: O(N^2)
```"""
    },
    "10": {
        "python": """```python
def max_product(self, nums: list[int]) -> int:
    if not nums: return 0
    max_val = min_val = result = nums[0]
    
    for i in range(1, len(nums)):
        # If current is negative, max and min will swap roles
        if nums[i] < 0:
            max_val, min_val = min_val, max_val
            
        max_val = max(nums[i], max_val * nums[i])
        min_val = min(nums[i], min_val * nums[i])
        result = max(result, max_val)
        
    return result
# Time Complexity: O(N)
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int MaxProduct(int[] nums) {
    if (nums == null || nums.Length == 0) return 0;
    int maxVal = nums[0], minVal = nums[0], result = nums[0];
    
    for (int i = 1; i < nums.Length; i++) {
        // If current is negative, max and min will swap roles
        if (nums[i] < 0) {
            int temp = maxVal; 
            maxVal = minVal; 
            minVal = temp;
        }
        maxVal = Math.Max(nums[i], maxVal * nums[i]);
        minVal = Math.Min(nums[i], minVal * nums[i]);
        result = Math.Max(result, maxVal);
    }
    return result;
}
// Time Complexity: O(N)
// Space Complexity: O(1)
```"""
    },
    "11": {
        "python": """```python
def find_median_sorted_arrays(self, A: list[int], B: list[int]) -> float:
    if len(A) > len(B): return self.find_median_sorted_arrays(B, A) # ensure A is smaller
    m, n = len(A), len(B)
    left, right = 0, m
    
    while left <= right:
        i = (left + right) // 2 # partition A
        j = (m + n + 1) // 2 - i # partition B
        
        max_left_a = float('-inf') if i == 0 else A[i - 1]
        min_right_a = float('inf') if i == m else A[i]
        max_left_b = float('-inf') if j == 0 else B[j - 1]
        min_right_b = float('inf') if j == n else B[j]
        
        if max_left_a <= min_right_b and max_left_b <= min_right_a:
            # Correct partition found
            if (m + n) % 2 == 0:
                return (max(max_left_a, max_left_b) + min(min_right_a, min_right_b)) / 2.0
            else:
                return max(max_left_a, max_left_b)
        elif max_left_a > min_right_b:
            right = i - 1 # move partition left in A
        else:
            left = i + 1 # move partition right in A
            
    return 0.0
# Time Complexity: O(log(min(M, N)))
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public double FindMedianSortedArrays(int[] A, int[] B) {
    if (A.Length > B.Length) return FindMedianSortedArrays(B, A); // ensure A is smaller
    int m = A.Length, n = B.Length;
    int left = 0, right = m;
    
    while (left <= right) {
        int i = (left + right) / 2; // partition A
        int j = (m + n + 1) / 2 - i; // partition B
        
        int maxLeftA = (i == 0) ? int.MinValue : A[i - 1];
        int minRightA = (i == m) ? int.MaxValue : A[i];
        int maxLeftB = (j == 0) ? int.MinValue : B[j - 1];
        int minRightB = (j == n) ? int.MaxValue : B[j];
        
        if (maxLeftA <= minRightB && maxLeftB <= minRightA) {
            // Correct partition found
            if ((m + n) % 2 == 0) {
                return (Math.Max(maxLeftA, maxLeftB) + Math.Min(minRightA, minRightB)) / 2.0;
            } else {
                return Math.Max(maxLeftA, maxLeftB);
            }
        } else if (maxLeftA > minRightB) {
            right = i - 1; // move partition left in A
        } else {
            left = i + 1; // move partition right in A
        }
    }
    return 0.0;
}
// Time Complexity: O(log(min(M, N)))
// Space Complexity: O(1)
```"""
    },
    "12": {
        "python": """```python
def trap(self, height: list[int]) -> int:
    if not height: return 0
    left, right = 0, len(height) - 1
    left_max = right_max = total_water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max: left_max = height[left]
            else: total_water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max: right_max = height[right]
            else: total_water += right_max - height[right]
            right -= 1
            
    return total_water
# Time Complexity: O(N)
# Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int Trap(int[] height) {
    if (height == null || height.Length == 0) return 0;
    int left = 0, right = height.Length - 1;
    int leftMax = 0, rightMax = 0, totalWater = 0;
    
    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) leftMax = height[left];
            else totalWater += leftMax - height[left];
            left++;
        } else {
            if (height[right] >= rightMax) rightMax = height[right];
            else totalWater += rightMax - height[right];
            right--;
        }
    }
    return totalWater;
}
// Time Complexity: O(N)
// Space Complexity: O(1)
```"""
    },
    "13": {
        "python": """```python
def daily_temperatures(self, temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    res = [0] * n
    stack = []
    
    for i in range(n):
        # While current temp is greater than temp at stack top
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_index = stack.pop()
            res[prev_index] = i - prev_index
        stack.append(i)
        
    return res
# Time Complexity: O(N)
# Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int[] DailyTemperatures(int[] temperatures) {
    int n = temperatures.Length;
    int[] res = new int[n];
    Stack<int> stack = new Stack<int>();
    
    for (int i = 0; i < n; i++) {
        // While current temp is greater than temp at stack top
        while (stack.Count > 0 && temperatures[i] > temperatures[stack.Peek()]) {
            int prevIndex = stack.Pop();
            res[prevIndex] = i - prevIndex;
        }
        stack.Push(i);
    }
    return res;
}
// Time Complexity: O(N)
// Space Complexity: O(N)
```"""
    },
    "14": {
        "python": """```python
def min_distance(self, word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] # No op
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], # Replace
                                   dp[i - 1][j],     # Delete
                                   dp[i][j - 1])     # Insert
                                   
    return dp[m][n]
# Time Complexity: O(M * N)
# Space Complexity: O(M * N)
```""",
        "csharp": """```csharp
public int MinDistance(string word1, string word2) {
    int m = word1.Length, n = word2.Length;
    int[][] dp = new int[m + 1][];
    for(int i=0; i<=m; i++) dp[i] = new int[n + 1];
    
    // Base cases
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1]; // No op
            } else {
                dp[i][j] = 1 + Math.Min(dp[i - 1][j - 1], // Replace
                               Math.Min(dp[i - 1][j],     // Delete
                                        dp[i][j - 1]));   // Insert
            }
        }
    }
    return dp[m][n];
}
// Time Complexity: O(M * N)
// Space Complexity: O(M * N)
```"""
    },
    "15": {
        "python": """```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache: return -1
        node = self.cache[key]
        self._remove(node)
        self._insert(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        if len(self.cache) == self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
            
        new_node = Node(key, value)
        self._insert(new_node)
        self.cache[key] = new_node

    def _remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert(self, node: Node) -> None:
        node.next = self.head.next
        node.next.prev = node
        self.head.next = node
        node.prev = self.head
# Time Complexity: O(1) for both get and put
# Space Complexity: O(Capacity)
```""",
        "csharp": """```csharp
public class LRUCache {
    class Node { 
        public int key, val; 
        public Node prev, next; 
    }
    private Dictionary<int, Node> map = new Dictionary<int, Node>();
    private int capacity;
    private Node head, tail;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        head = new Node(); 
        tail = new Node();
        head.next = tail; 
        tail.prev = head; // Connect dummy head and tail
    }
    
    public int Get(int key) {
        if (!map.ContainsKey(key)) return -1;
        Node node = map[key];
        Remove(node); // Move to head (MRU)
        Insert(node);
        return node.val;
    }
    
    public void Put(int key, int value) {
        if (map.ContainsKey(key)) {
            Remove(map[key]);
        }
        if (map.Count == capacity) {
            map.Remove(tail.prev.key);
            Remove(tail.prev); // Evict LRU
        }
        Node node = new Node(); 
        node.key = key; 
        node.val = value;
        Insert(node);
        map[key] = node;
    }
    
    private void Remove(Node node) {
        node.prev.next = node.next; 
        node.next.prev = node.prev;
    }
    
    private void Insert(Node node) { // Insert right after head
        node.next = head.next; 
        node.next.prev = node;
        head.next = node; 
        node.prev = head;
    }
}
// Time Complexity: O(1) for both get and put
// Space Complexity: O(Capacity)
```"""
    },
    "16": {
        "python": """```python
def maximal_rectangle(self, matrix: list[list[str]]) -> int:
    if not matrix or not matrix[0]: return 0
    cols = len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for row in matrix:
        # Update histogram heights
        for c in range(cols):
            heights[c] = heights[c] + 1 if row[c] == '1' else 0
        max_area = max(max_area, self._max_histogram(heights))
        
    return max_area

def _max_histogram(self, heights: list[int]) -> int:
    stack = []
    max_val = 0
    n = len(heights)
    
    for i in range(n + 1):
        h = 0 if i == n else heights[i]
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_val = max(max_val, height * width)
        stack.append(i)
        
    return max_val
# Time Complexity: O(R * C)
# Space Complexity: O(C)
```""",
        "csharp": """```csharp
public int MaximalRectangle(char[][] matrix) {
    if (matrix == null || matrix.Length == 0) return 0;
    int cols = matrix[0].Length;
    int[] heights = new int[cols];
    int maxArea = 0;
    
    foreach (char[] row in matrix) {
        // Update histogram heights
        for (int c = 0; c < cols; c++) {
            heights[c] = (row[c] == '1') ? heights[c] + 1 : 0;
        }
        maxArea = Math.Max(maxArea, MaxHistogram(heights));
    }
    return maxArea;
}

private int MaxHistogram(int[] heights) {
    Stack<int> stack = new Stack<int>();
    int max = 0, n = heights.Length;
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];
        while (stack.Count > 0 && h < heights[stack.Peek()]) {
            int height = heights[stack.Pop()];
            int width = stack.Count == 0 ? i : i - stack.Peek() - 1;
            max = Math.Max(max, height * width);
        }
        stack.Push(i);
    }
    return max;
}
// Time Complexity: O(R * C)
// Space Complexity: O(C)
```"""
    },
    "17": {
        "python": """```python
def ladder_length(self, begin_word: str, end_word: str, word_list: list[str]) -> int:
    word_set = set(word_list)
    if end_word not in word_set: return 0
    
    from collections import deque
    queue = deque([begin_word])
    level = 1
    
    while queue:
        for _ in range(len(queue)): # Level-by-level processing
            curr = queue.popleft()
            for j in range(len(curr)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == curr[j]: continue
                    next_word = curr[:j] + c + curr[j+1:]
                    if next_word == end_word: return level + 1
                    if next_word in word_set: # remove serves as 'visited' check
                        word_set.remove(next_word)
                        queue.append(next_word)
        level += 1
        
    return 0
# Time Complexity: O(M^2 * N) where M is word length, N is number of words
# Space Complexity: O(M * N)
```""",
        "csharp": """```csharp
public int LadderLength(string beginWord, string endWord, IList<string> wordList) {
    HashSet<string> set = new HashSet<string>(wordList);
    if (!set.Contains(endWord)) return 0;
    
    Queue<string> queue = new Queue<string>();
    queue.Enqueue(beginWord);
    int level = 1;
    
    while (queue.Count > 0) {
        int size = queue.Count;
        for (int i = 0; i < size; i++) { // Level-by-level processing
            string curr = queue.Dequeue();
            char[] chars = curr.ToCharArray();
            for (int j = 0; j < chars.Length; j++) {
                char orig = chars[j];
                for (char c = 'a'; c <= 'z'; c++) { // Try all mutations
                    if (c == orig) continue;
                    chars[j] = c;
                    string next = new string(chars);
                    if (next.Equals(endWord)) return level + 1;
                    if (set.Remove(next)) { // remove serves as 'visited' check
                        queue.Enqueue(next);
                    }
                }
                chars[j] = orig; // Backtrack
            }
        }
        level++;
    }
    return 0;
}
// Time Complexity: O(M^2 * N) where M is word length, N is number of words
// Space Complexity: O(M * N)
```"""
    }
}

base_dir = r"C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\13-optimization-dp\snippets"
for k, v in translations.items():
    with open(os.path.join(base_dir, "python", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["python"])
    with open(os.path.join(base_dir, "csharp", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["csharp"])

print("Translations for chapter 13 partly applied!")
