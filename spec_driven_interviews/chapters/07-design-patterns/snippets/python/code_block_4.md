```python
import threading

class LedgerConnectionPool:
    _instance = None # <1>
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None: # <2>
            with cls._lock: # <3>
                if cls._instance is None: # <4>
                    cls._instance = super(LedgerConnectionPool, cls).__new__(cls) # <5>
        return cls._instance
```