```java
public class Order {
    public enum Side { BUY, SELL }
    private final String id;
    private final String instrumentId;
    private final Side side;
    private final long price; // Fixed-point integer (smallest atomic unit)
    private final long quantity;

    public Order(String id, String instrumentId, Side side, long price, long quantity) {
        if (price <= 0) throw new IllegalArgumentException("Price must be positive");
        if (quantity <= 0) throw new IllegalArgumentException("Quantity must be positive");
        this.id = id;
        this.instrumentId = instrumentId;
        this.side = side;
        this.price = price;
        this.quantity = quantity;
    }

    public String getId() { return id; }
    public String getInstrumentId() { return instrumentId; }
    public Side getSide() { return side; }
    public long getPrice() { return price; }
    public long getQuantity() { return quantity; }
}
```