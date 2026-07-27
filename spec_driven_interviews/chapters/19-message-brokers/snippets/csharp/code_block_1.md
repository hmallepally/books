```csharp
using Confluent.Kafka;
using System.Threading.Tasks;

public class TransactionEventProducer 
{
    private final IProducer<string, string> _producer;
    private final string _topic;

    public TransactionEventProducer(string bootstrapServers, string topic) 
    {
        var config = new ProducerConfig
        {
            BootstrapServers = bootstrapServers,
            EnableIdempotence = true,
            Acks = Acks.All
        };
        _producer = new ProducerBuilder<string, string>(config).Build();
        _topic = topic;
    }

    public async Task PublishEventAsync(string accountId, string eventJson) 
    {
        // Shard by accountId to guarantee partition message ordering
        var message = new Message<string, string> { Key = accountId, Value = eventJson };
        await _producer.ProduceAsync(_topic, message);
    }
}
```