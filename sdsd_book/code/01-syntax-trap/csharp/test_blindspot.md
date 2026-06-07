```csharp
// tests/TransactionsControllerTests.cs (AI-Generated)
using Xunit;

public class TransactionsControllerTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;

    public TransactionsControllerTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
    }

    [Fact]
    public async Task Test_Transaction_Caching()
    {
        var client = _factory.CreateClient();
        var request = new HttpRequestMessage(HttpMethod.Get, "/api/v1/transactions?limit=10");
        request.Headers.Add("Authorization", "Bearer mock_token");

        // First request: Cache Miss
        var response1 = await client.SendAsync(request);
        response1.EnsureSuccessStatusCode();
        Assert.Equal(1, _factory.MockRedisDb.StringGetCallCount);
        Assert.Equal(1, _factory.MockRedisDb.StringSetCallCount);

        // Second request: Cache Hit
        var request2 = new HttpRequestMessage(HttpMethod.Get, "/api/v1/transactions?limit=10");
        request2.Headers.Add("Authorization", "Bearer mock_token");
        var response2 = await client.SendAsync(request2);
        response2.EnsureSuccessStatusCode();
        Assert.Equal(2, _factory.MockRedisDb.StringGetCallCount);
        // Verify set was not called again (cache hit)
        Assert.Equal(1, _factory.MockRedisDb.StringSetCallCount);
    }
}
```