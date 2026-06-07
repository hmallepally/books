```csharp
// Threat Modeling Config
public void ConfigureServices(IServiceCollection services)
{
    services.AddAuthorization(options => {
        options.FallbackPolicy = new AuthorizationPolicyBuilder()
            .RequireAuthenticatedUser()
            .Build();
    });
}
```