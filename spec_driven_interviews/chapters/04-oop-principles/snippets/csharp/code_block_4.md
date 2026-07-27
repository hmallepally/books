```csharp
public class SettlementProcessor 
{
    private readonly List<ISettlementRoute> _routes;

    public SettlementProcessor(List<ISettlementRoute> routes) 
    {
        _routes = routes;
    }

    public void Execute(TransactionRecord transaction) 
    {
        var activeRoute = _routes
            .FirstOrDefault(route => route.Supports(transaction))
            ?? throw new NoRouteFoundException("No supported route found");
            
        activeRoute.Process(transaction);
    }
}
```
