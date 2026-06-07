```python
# tests/test_transactions.py (AI-Generated)
import pytest
from httpx import AsyncClient
@pytest.mark.asyncio
async def test_transaction_caching(test_app, mock_redis, auth_headers):
    # First request: Cache Miss
    response1 = await test_app.get(
        "/api/v1/transactions?limit=10", 
        headers=auth_headers
    )
    assert response1.status_code == 200
    assert mock_redis.get_call_count == 1
    assert mock_redis.set_call_count == 1
    # Second request: Cache Hit
    response2 = await test_app.get(
        "/api/v1/transactions?limit=10", 
        headers=auth_headers
    )
    assert response2.status_code == 200
    assert mock_redis.get_call_count == 2
    # Verify set was not called again (cache hit)
    assert mock_redis.set_call_count == 1 
```