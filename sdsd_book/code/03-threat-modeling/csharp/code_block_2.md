```csharp
// Threat Model Test
[Fact]
public async Task Test_Authentication_Required()
{
    var response = await _unauthClient.GetAsync("/api/v1/transactions");
    Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
}
```