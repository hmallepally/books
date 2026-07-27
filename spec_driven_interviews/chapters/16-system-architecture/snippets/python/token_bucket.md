```python
import time
import threading

class TokenBucket:
    def __init__(self, max_tokens: int, refill_rate_per_second: int):
        self.max_tokens = max_tokens
        self.refill_rate_per_second = refill_rate_per_second
        self.tokens = max_tokens
        self.timestamp_nanos = time.monotonic_ns()
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        with self.lock:
            now = time.monotonic_ns()
            elapsed = now - self.timestamp_nanos
            refilled = min(self.max_tokens, 
                self.tokens + elapsed * self.refill_rate_per_second // 1_000_000_000)
            
            if refilled <= 0:
                return False
                
            self.tokens = refilled - 1
            self.timestamp_nanos = now
            return True
```