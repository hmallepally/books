```python
import threading

class LedgerConnectionPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None: # First check (no lock)
            with cls._lock:
                if cls._instance is None: # Second check (with lock)
                    cls._instance = super(LedgerConnectionPool, cls).__new__(cls)
        return cls._instance
```