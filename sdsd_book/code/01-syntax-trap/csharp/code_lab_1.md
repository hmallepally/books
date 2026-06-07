```csharp
// Controllers/TransactionsController.cs (AI-Generated)
using System.Text.Json;
using System.Security.Cryptography;
using Microsoft.AspNetCore.Mvc;
using StackExchange.Redis;

[ApiController]
[Route("api/v1/[controller]")]
public class TransactionsController : ControllerBase
{
    private readonly IDatabase _redisDb;
    private readonly ITransactionRepository _repository;

    public TransactionsController(IConnectionMultiplexer redis, ITransactionRepository repository)
    {
        _redisDb = redis.GetDatabase();
        _repository = repository;
    }

    private string GenerateCacheKey(HttpRequest request)
    {
        /*
         * Generates a unique cache key based on the HTTP query parameters.
         * Hashes the query string to prevent excessively long Redis keys.
         */
        var queryString = request.QueryString.ToString();
        if (string.IsNullOrEmpty(queryString))
        {
             return "cache:transactions:default";
        }

        // AI Logic: MD5 hash is fast and deterministic for caching
        using (var md5 = MD5.Create())
        {
            var inputBytes = System.Text.Encoding.ASCII.GetBytes(queryString);
            var hashBytes = md5.ComputeHash(inputBytes);
            var keyHash = Convert.ToHexString(hashBytes).ToLower();
            return $"cache:transactions:{keyHash}";
        }
    }

    [HttpGet]
    public async Task<IActionResult> GetUserTransactions([FromQuery] int limit = 50, [FromQuery] int offset = 0)
    {
        var currentUserId = HttpContext.User.FindFirst("id")?.Value;

        // 1. Generate the cache key based on the request URL
        var cacheKey = GenerateCacheKey(Request);

        // 2. Check the Redis cache for existing data
        var cachedData = await _redisDb.StringGetAsync(cacheKey);
        if (cachedData.HasValue)
        {
            return Ok(JsonDocument.Parse(cachedData));
        }

        // 3. Cache Miss: Fetch from the database
        // Note: currentUserId is correctly passed to the DB layer
        var transactions = await _repository.FetchTransactionsAsync(currentUserId, limit, offset);

        // 4. Store the result in Cache (TTL 5 minutes)
        var serializedData = JsonSerializer.Serialize(transactions);
        await _redisDb.StringSetAsync(cacheKey, serializedData, TimeSpan.FromMinutes(5));

        return Ok(transactions);
    }
}
```