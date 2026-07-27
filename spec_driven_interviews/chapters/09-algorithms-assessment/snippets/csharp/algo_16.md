```csharp
public int[] FindOrder(int numCourses, int[][] prerequisites)
{
    var inDegree = new int[numCourses];
    var adj = new List<List<int>>();
    for (int i = 0; i < numCourses; i++) adj.Add(new List<int>());
    foreach (int[] p in prerequisites)
    {
        adj[p[1]].Add(p[0]);
        inDegree[p[0]]++;
    }

    var queue = new Queue<int>();
    for (int i = 0; i < numCourses; i++) if (inDegree[i] == 0) queue.Enqueue(i);

    int[] order = new int[numCourses];
    int idx = 0;
    while (queue.Count > 0)
    {
        int curr = queue.Dequeue();
        order[idx++] = curr;
        foreach (int neighbor in adj[curr])
        {
            if (--inDegree[neighbor] == 0) queue.Enqueue(neighbor);
        }
    }
    return idx == numCourses ? order : new int[0];
}
```
