```python
class UserService:
    def __init__(self, db_repository, redis_client):
        self.db_repository = db_repository
        self.redis_client = redis_client

    def get_user(self, user_id: str):
        cache_key = f"user:{user_id}"
        user = self.redis_client.get(cache_key)
        
        if user is None:
            # Cache miss: read from DB
            user = self.db_repository.find_by_id(user_id)
            if user is None:
                raise ValueError("User not found")
            # Populate cache
            self.redis_client.set(cache_key, user)
            
        return user
```