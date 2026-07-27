```csharp
public class LedgerConnectionPool 
{
    private static volatile LedgerConnectionPool _instance;
    private static readonly object _lock = new object();
    
    private LedgerConnectionPool() {}
    
    public static LedgerConnectionPool Instance 
    {
        get 
        {
            if (_instance == null) // First check (no lock)
            {
                lock (_lock)
                {
                    if (_instance == null) // Second check (with lock)
                    {
                        _instance = new LedgerConnectionPool();
                    }
                }
            }
            return _instance;
        }
    }
}
```