import os
import subprocess
import typer

def detect_framework(project_root="."):
    """Detects the test framework based on project marker files."""
    # Java Maven
    if os.path.exists(os.path.join(project_root, "pom.xml")):
        return "maven"
    
    # Java Gradle
    if os.path.exists(os.path.join(project_root, "build.gradle")) or os.path.exists(os.path.join(project_root, "build.gradle.kts")):
        return "gradle"
    
    # Python
    if os.path.exists(os.path.join(project_root, "pyproject.toml")) or \
       os.path.exists(os.path.join(project_root, "pytest.ini")) or \
       os.path.exists(os.path.join(project_root, "requirements.txt")):
        return "python"
        
    # C# / .NET
    for file in os.listdir(project_root):
        if file.endswith(".csproj") or file.endswith(".sln"):
            return "dotnet"
            
    return "unknown"

def execute_tests(framework, project_root=".", skip_mutation=False):
    """Executes the test suite for the detected framework."""
    env = os.environ.copy()
    
    # Allow manual override via .env
    override = env.get("SDSD_TEST_FRAMEWORK")
    if override:
        framework = override
        typer.echo(f"Using forced test framework override from .env: {framework}")

    # Enforce Mutation Testing for Java (Gradle)
    if framework == "gradle" and not skip_mutation:
        build_file_path = os.path.join(project_root, "build.gradle")
        if os.path.exists(build_file_path):
            with open(build_file_path, "r") as f:
                content = f.read()
                if "info.solidsoft.pitest" not in content:
                    typer.secho("\n[FATAL] Mutation tests (PIT) are not configured in build.gradle.", fg=typer.colors.RED)
                    typer.secho("SDSD methodology mandates mutation coverage. Add the 'info.solidsoft.pitest' plugin.", fg=typer.colors.YELLOW)
                    typer.echo("To suppress this check, run: sdsd verify --skip-mutation")
                    raise typer.Exit(code=1)

    # Detect gradlew in project root
    gradle_bin = "gradle"
    if os.path.exists(os.path.join(project_root, "gradlew.bat")):
        gradle_bin = "gradlew.bat"
    elif os.path.exists(os.path.join(project_root, "gradlew")):
        gradle_bin = "./gradlew"

    commands = {
        "maven": ["mvn", "test"],
        "gradle": [gradle_bin, "test", "pitest"],
        "python": ["pytest", "-v"],
        "dotnet": ["dotnet", "test"]
    }
    
    if framework not in commands:
        typer.secho(f"Error: Unsupported or unknown test framework '{framework}'.", fg=typer.colors.RED)
        typer.echo("Please configure SDSD_TEST_FRAMEWORK in your .env file.")
        raise typer.Exit(code=1)
        
    cmd = commands[framework]
    typer.secho(f"Running tests via: {' '.join(cmd)}", fg=typer.colors.CYAN)
    
    try:
        use_shell = (os.name == "nt")
        result = subprocess.run(cmd, cwd=project_root, env=env, shell=use_shell)
        if result.returncode != 0:
            typer.secho(f"\n[FATAL] SDSD Verification Failed! Tests returned non-zero exit code: {result.returncode}", fg=typer.colors.RED)
            typer.secho("The implemented code violates the defined test invariants.", fg=typer.colors.RED)
            raise typer.Exit(code=result.returncode)
        
        typer.secho("\n[SUCCESS] SDSD Verification Passed! All invariants satisfied.", fg=typer.colors.GREEN)
        
    except FileNotFoundError:
        typer.secho(f"Error: Could not find the executable for '{cmd[0]}'. Please check your PATH or .env configuration.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
