```python
# Anti-pattern: Inspecting properties to determine routing
if tx.amount > LIMIT:
    fed_wire_route.process(tx)
else:
    ach_route.process(tx)
```
