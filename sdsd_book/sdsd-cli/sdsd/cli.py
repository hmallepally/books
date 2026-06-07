"""SDSD CLI — Main entry point.

Spec-Driven Secure Development command-line interface.
Assemble secure, context-rich prompts for AI coding agents.

Usage:
    sdsd init                          Scaffold .agent/ directory
    sdsd prompt create --type feature  Assemble a prompt
    sdsd validate                      Check SDSD readiness
"""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from sdsd import __version__

from sdsd.commands.init import init_command
from sdsd.commands.prompt import prompt_app
from sdsd.commands.validate import validate_command
from sdsd.commands import verify

# Load environment variables from .env if present
load_dotenv()

app = typer.Typer(
    name="sdsd",
    help=(
        "SDSD CLI -- Spec-Driven Secure Development\n\n"
        "Assemble secure, context-rich prompts for AI coding agents.\n"
        "Based on the SDSD methodology by Harinath Mallepally."
    ),
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"sdsd-cli v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        None, "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """SDSD CLI — Spec-Driven Secure Development."""


# Register commands
app.command("init", help="Scaffold the .agent/ directory structure.")(init_command)
app.add_typer(prompt_app, name="prompt", help="Prompt assembly commands.")
app.command("validate", help="Check SDSD readiness of a project.")(validate_command)
app.add_typer(verify.app, name="verify")


if __name__ == "__main__":
    app()
