```java
public int[] findOrder(int numCourses, int[][] prerequisites) {
    var inDegree = new int[numCourses];
    var adj = new ArrayList<List<Integer>>();
    for (int i = 0; i < numCourses; i++) adj.add(new ArrayList<>());
    
    for (int[] p : prerequisites) {
        adj.get(p[1]).add(p[0]);
        inDegree[p[0]]++;
    }
    
    Queue<Integer> q = new ArrayDeque<>();
    for (int i = 0; i < numCourses; i++) {
        if (inDegree[i] == 0) q.offer(i);
    }
    
    int[] res = new int[numCourses];
    int idx = 0;
    while (!q.isEmpty()) {
        int curr = q.poll();
        res[idx++] = curr;
        for (int next : adj.get(curr)) {
            if (--inDegree[next] == 0) q.offer(next);
        }
    }
    return idx == numCourses ? res : new int[0]; // If not all courses taken, cycle exists
}
// Time Complexity: O(V + E)
// Space Complexity: O(V + E)
```
