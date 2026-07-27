```csharp
public class Order 
{
    public enum OrderSide { Buy, Sell }
    public string Id { get; }
    public string InstrumentId { get; }
    public OrderSide Side { get; }
    public long Price { get; } // Fixed-point integer
    public long Quantity { get; }

    public Order(string id, string instrumentId, OrderSide side, long price, long quantity) 
    {
        if (price <= 0) throw new ArgumentException("Price must be positive");
        if (quantity <= 0) throw new ArgumentException("Quantity must be positive");
        Id = id;
        InstrumentId = instrumentId;
        Side = side;
        Price = price;
        Quantity = quantity;
    }
}
```