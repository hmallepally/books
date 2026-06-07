```python
# SECURE IMPLEMENTATION
ALLOWED_SORT_COLUMNS = {"amount", "created_at", "status"}
@router.get("/admin/transactions/report")
async def get_report(
    sort_by: str, 
    admin_org_id: str = Depends(get_admin_org),
    db: Session = Depends(get_db)
):
    # THREAT MITIGATION: SQLi Whitelist Validation
    if sort_by not in ALLOWED_SORT_COLUMNS:
        raise HTTPException(status_code=400, detail="Invalid sort parameter.")
    # INVARIANT: Tenant Isolation via SQLAlchemy ORM (No raw SQL)
    # The ORM automatically escapes all inputs, preventing SQLi.
    query = select(Transaction).where(
        Transaction.org_id == admin_org_id
    ).order_by(
        desc(getattr(Transaction, sort_by))
    )
    result = await db.execute(query)
    return result.scalars().all()
```