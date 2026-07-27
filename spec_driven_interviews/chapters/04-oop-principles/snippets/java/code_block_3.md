```java
// Anti-pattern: Inspecting properties to determine routing
if (tx.getAmount().compareTo(LIMIT) > 0) {
    fedWireRoute.process(tx);
} else {
    achRoute.process(tx);
}
```
