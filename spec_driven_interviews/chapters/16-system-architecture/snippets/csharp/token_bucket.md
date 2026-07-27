```csharp
using System;
using System.Diagnostics;
using System.Threading;

public class TokenBucket 
{
    private class State 
    {
        public long Tokens { get; }
        public long TimestampNanos { get; }

        public State(long tokens, long timestampNanos) 
        {
            Tokens = tokens;
            TimestampNanos = timestampNanos;
        }
    }
    
    private State _state;
    private readonly long _maxTokens;
    private readonly long _refillRatePerSecond;

    public TokenBucket(long maxTokens, long refillRatePerSecond) 
    {
        _maxTokens = maxTokens;
        _refillRatePerSecond = refillRatePerSecond;
        _state = new State(maxTokens, Stopwatch.GetTimestamp());
    }

    public bool AllowRequest() 
    {
        while (true) 
        {
            State current = Volatile.Read(ref _state);
            long now = Stopwatch.GetTimestamp();
            long elapsed = now - current.TimestampNanos;
            
            long refilled = Math.Min(_maxTokens,
                current.Tokens + elapsed * _refillRatePerSecond / 1_000_000_000L);
                
            if (refilled <= 0) return false;
            
            State next = new State(refilled - 1, now);
            if (Interlocked.CompareExchange(ref _state, next, current) == current) 
            {
                return true;
            }
        }
    }
}
```