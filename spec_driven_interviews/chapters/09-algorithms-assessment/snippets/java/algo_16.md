```java
public int[] findOrder(int numCourses, int[][] prerequisites) {
    var inDegree = new int[numCourses];
    var adj = new ArrayList<List<Integer>>();
    for (int i = 0; i < numCourses; i++) adj.add(new ArrayList<>());
    for (int[] p : prerequisites) {
        adj.get(p[1]).add(p[0]);
        inDegree[p[0]]++;
    }

    var queue = new ArrayDeque<Integer>();
    for (int i = 0; i < numCourses; i++) if (inDegree[i] == 0) queue.offer(i);

    int[] order = new int[numCourses];
    int idx = 0;
    while (!queue.isEmpty()) {
        int curr = queue.poll();
        order[idx++] = curr;
        for (int neighbor : adj.get(curr)) {
            if (--inDegree[neighbor] == 0) queue.offer(neighbor);
        }
    }
    return idx == numCourses ? order : new int[0];
}
```
