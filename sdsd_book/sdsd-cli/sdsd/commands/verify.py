import typer
import os
from sdsd.core.runner import detect_framework, execute_tests

app = typer.Typer(help="Verify implementation against SDSD invariant tests.")

@app.callback(invoke_without_command=True)
def verify(
    path: str = typer.Option(".", "--path", "-p", help="Path to the project root directory."),
    skip_mutation: bool = typer.Option(False, "--skip-mutation", help="Suppress mutation testing requirement."),
):
    """
    Auto-detects the project framework and runs the test suite. 
    Strictly enforces that all invariant tests pass.
    """
    typer.secho("SDSD Verification Engine Initializing...", fg=typer.colors.MAGENTA, bold=True)
    
    if not os.path.exists(path):
        typer.secho(f"Error: Path '{path}' does not exist.", fg=typer.colors.RED)
        raise typer.Exit(1)
        
    framework = detect_framework(path)
    if framework == "unknown" and not os.environ.get("SDSD_TEST_FRAMEWORK"):
        typer.secho(f"Could not automatically detect test framework in '{path}'.", fg=typer.colors.YELLOW)
        typer.echo("Please ensure marker files (pom.xml, build.gradle, pyproject.toml, .csproj) exist,")
        typer.echo("OR set SDSD_TEST_FRAMEWORK in your .env file.")
        raise typer.Exit(1)
        
    if framework != "unknown":
        typer.echo(f"Detected project framework: ")
        typer.secho(f"{framework.upper()}", fg=typer.colors.GREEN, bold=True)
        
    execute_tests(framework, path, skip_mutation)
