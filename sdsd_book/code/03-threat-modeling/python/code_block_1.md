```python
# WARNING: VULNERABLE CODE
@router.get("/admin/transactions/report")
async def get_report(sort_by: str, db: Session = Depends(get_db)):
    # The AI uses an f-string for dynamic column sorting
    query = f"SELECT * FROM transactions ORDER BY {sort_by} DESC"
    result = await db.execute(query)
    return result.fetchall()
```