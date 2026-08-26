```java
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class TokenBucket {
    private record State(long tokens, long timestampNanos) {}
    
    private final AtomicReference<State> state;
    private final long maxTokens;
    private final long refillRatePerSecond;

    public TokenBucket(long maxTokens, long refillRatePerSecond) {
        this.maxTokens = maxTokens;
        this.refillRatePerSecond = refillRatePerSecond;
        this.state = new AtomicReference<>(new State(maxTokens, System.nanoTime()));
    }

    public boolean allowRequest() {
        while (true) {
            State current = state.get();
            long now = System.nanoTime();
            long elapsed = now - current.timestampNanos();
            long refilled = Math.min(maxTokens,
                current.tokens() + elapsed * refillRatePerSecond / 1_000_000_000L);
            if (refilled <= 0) return false;
            State next = new State(refilled - 1, now);
            if (state.compareAndSet(current, next)) return true;
        }
    }
}
```