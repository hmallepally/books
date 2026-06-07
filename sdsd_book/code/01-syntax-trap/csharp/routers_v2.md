```csharp
private string GenerateCacheKey(HttpRequest request, string userId)
{
    /* Generates a tenant-isolated cache key. */
    var queryString = request.QueryString.ToString();
    var keyHash = ComputeMd5Hash(queryString);
    // The cache key is now rigidly bound to the tenant ID
    return $"cache:tenant:{userId}:transactions:{keyHash}";
}
```