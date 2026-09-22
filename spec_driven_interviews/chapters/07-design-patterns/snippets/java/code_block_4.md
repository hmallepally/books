```java
public class LedgerConnectionPool {
    private static volatile LedgerConnectionPool instance; // <1>
    
    private LedgerConnectionPool() {
        // Prevent reflection instantiation
        if (instance != null) {
            throw new IllegalStateException("Already instantiated");
        }
    }
    
    public static LedgerConnectionPool getInstance() {
        if (instance == null) { // <2>
            synchronized (LedgerConnectionPool.class) { // <3>
                if (instance == null) { // <4>
                    instance = new LedgerConnectionPool(); // <5>
                }
            }
        }
        return instance;
    }
}
```