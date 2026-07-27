```csharp
using System.Threading.Tasks;
using Microsoft.Extensions.Caching.Distributed;
using System.Text.Json;
using System.Collections.Generic;

public class UserService 
{
    private readonly IUserRepository _dbRepository;
    private readonly IDistributedCache _cache;

    public UserService(IUserRepository dbRepository, IDistributedCache cache) 
    {
        _dbRepository = dbRepository;
        _cache = cache;
    }

    public User GetUser(string userId) 
    {
        string cacheKey = $"user:{userId}";
        string cachedUser = _cache.GetString(cacheKey);
        
        if (cachedUser != null) 
        {
            return JsonSerializer.Deserialize<User>(cachedUser);
        }

        // Cache miss: read from DB
        User user = _dbRepository.FindById(userId);
        if (user == null) 
        {
            throw new KeyNotFoundException();
        }
        
        // Populate cache
        _cache.SetString(cacheKey, JsonSerializer.Serialize(user));
        return user;
    }
}
```