```csharp
public string GenerateCacheKey(HttpRequest request)
{
    var queryString = request.QueryString.ToString();
    var keyHash = ComputeMd5Hash(queryString);
    return $"cache:transactions:{keyHash}";
}
```