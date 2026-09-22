```csharp
public class LedgerConnectionPool 
{
    private static volatile LedgerConnectionPool _instance; // <1>
    private static readonly object _lock = new object();
    
    private LedgerConnectionPool() {}
    
    public static LedgerConnectionPool Instance 
    {
        get 
        {
            if (_instance == null) // <2>
            {
                lock (_lock) // <3>
                {
                    if (_instance == null) // <4>
                    {
                        _instance = new LedgerConnectionPool(); // <5>
                    }
                }
            }
            return _instance;
        }
    }
}
```