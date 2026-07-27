```csharp
public class Trie {
    private readonly TrieNode _root = new();

    private class TrieNode {
        public TrieNode?[] Children = new TrieNode?[26];
        public bool IsEnd = false;
    }

    public void Insert(string word) {
        var node = _root;
        foreach (char c in word) {
            int idx = c - 'a';
            node.Children[idx] ??= new TrieNode();
            node = node.Children[idx]!;
        }
        node.IsEnd = true;
    }

    public bool Search(string word) {
        var node = FindNode(word);
        return node is { IsEnd: true };
    }

    public bool StartsWith(string prefix) {
        return FindNode(prefix) != null;
    }

    private TrieNode? FindNode(string s) {
        var node = _root;
        foreach (char c in s) {
            int idx = c - 'a';
            if (node.Children[idx] == null) return null;
            node = node.Children[idx]!;
        }
        return node;
    }
}
```
