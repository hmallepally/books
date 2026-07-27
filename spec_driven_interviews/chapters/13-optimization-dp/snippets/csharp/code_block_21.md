```csharp
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
```