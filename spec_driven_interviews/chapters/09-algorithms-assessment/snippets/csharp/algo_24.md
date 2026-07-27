```csharp
public class TrieNode
{
    public TrieNode[] Children = new TrieNode[26];
    public bool IsWord = false;
}

public class Trie
{
    private TrieNode root = new TrieNode();

    public void Insert(string word)
    {
        TrieNode curr = root;
        foreach (char c in word)
        {
            int idx = c - 'a';
            if (curr.Children[idx] == null) curr.Children[idx] = new TrieNode();
            curr = curr.Children[idx];
        }
        curr.IsWord = true;
    }

    public bool Search(string word)
    {
        TrieNode node = GetNode(word);
        return node != null && node.IsWord;
    }

    public bool StartsWith(string prefix)
    {
        return GetNode(prefix) != null;
    }

    private TrieNode GetNode(string str)
    {
        TrieNode curr = root;
        foreach (char c in str)
        {
            int idx = c - 'a';
            if (curr.Children[idx] == null) return null;
            curr = curr.Children[idx];
        }
        return curr;
    }
}
```
