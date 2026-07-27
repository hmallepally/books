```csharp
// Anti-pattern: Inspecting properties to determine routing
if (tx.Amount > LIMIT) {
    fedWireRoute.Process(tx);
} else {
    achRoute.Process(tx);
}
```
