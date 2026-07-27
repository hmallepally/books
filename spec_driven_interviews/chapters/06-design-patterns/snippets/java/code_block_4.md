```java
public class LedgerConnectionPool {
    private static volatile LedgerConnectionPool instance;
    
    private LedgerConnectionPool() {
        // Prevent reflection instantiation
        if (instance != null) {
            throw new IllegalStateException("Already instantiated");
        }
    }
    
    public static LedgerConnectionPool getInstance() {
        if (instance == null) { // First check (no lock)
            synchronized (LedgerConnectionPool.class) {
                if (instance == null) { // Second check (with lock)
                    instance = new LedgerConnectionPool();
                }
            }
        }
        return instance;
    }
}
```