```python
class SettlementProcessor:
    def __init__(self, routes: list[SettlementRoute]):
        self._routes = routes

    def execute(self, transaction: TransactionRecord) -> None:
        active_route = next(
            (route for route in self._routes if route.supports(transaction)), 
            None
        )
        if not active_route:
            raise NoRouteFoundException("No supported route found")
            
        active_route.process(transaction)
```
