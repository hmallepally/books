"""AST-based dependency and signature crawler.

Performs the "Dependency Crawling" step described in Chapter 7, §7.2:
analyzes a target directory and automatically bundles the function
signatures of all internal utilities imported by that directory.

Supports:
- Python: Full AST parsing via the `ast` module
- Java:   Regex-based class/method signature extraction
- C#:     Regex-based class/method signature extraction
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from sdsd.models.spec import UtilitySignature


def detect_language(target: Path) -> str:
    """Detect the primary language of a project directory.

    Scans the target path (or its parent directory if a file) for
    source files and returns the dominant language.
    """
    scan_dir = target if target.is_dir() else target.parent

    counts = {"python": 0, "java": 0, "csharp": 0}
    for f in scan_dir.rglob("*"):
        if f.suffix == ".py":
            counts["python"] += 1
        elif f.suffix == ".java":
            counts["java"] += 1
        elif f.suffix == ".cs":
            counts["csharp"] += 1

    if not any(counts.values()):
        return "python"  # Default

    return max(counts, key=counts.get)


def crawl_target(target: Path, language: Optional[str] = None) -> list[UtilitySignature]:
    """Crawl a target path and extract utility signatures.

    This is the core function that implements the "Dependency Crawling"
    described in Ch.7, §7.2. It extracts function/class signatures
    that the AI needs as context to avoid hallucinating utilities.

    Args:
        target: Path to a file or directory to crawl.
        language: Override language detection ('python', 'java', 'csharp').

    Returns:
        List of discovered utility signatures.
    """
    if not target.exists():
        return []

    if language is None:
        language = detect_language(target)

    crawlers = {
        "python": _crawl_python,
        "java": _crawl_java,
        "csharp": _crawl_csharp,
    }

    crawler = crawlers.get(language, _crawl_python)

    if target.is_file():
        return crawler(target)

    signatures = []
    extensions = {
        "python": ".py",
        "java": ".java",
        "csharp": ".cs",
    }
    ext = extensions.get(language, ".py")

    for source_file in sorted(target.rglob(f"*{ext}")):
        # Skip test files and __pycache__
        rel = str(source_file.relative_to(target))
        if any(skip in rel for skip in ["__pycache__", "test_", "_test.", "tests/"]):
            continue
        signatures.extend(crawler(source_file))

    return signatures


# ---------------------------------------------------------------------------
# Python: Full AST-based crawling
# ---------------------------------------------------------------------------

def _crawl_python(filepath: Path) -> list[UtilitySignature]:
    """Extract function and class signatures from a Python file using AST."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    module_path = _path_to_module(filepath)
    signatures = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            sig = _extract_function_signature(node)
            signatures.append(UtilitySignature(
                module_path=module_path,
                name=node.name,
                kind="function",
                signature=sig,
            ))
        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("_"):
                        methods.append(item.name)

            # Build class-level signature
            init_node = None
            for item in ast.iter_child_nodes(node):
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    init_node = item
                    break

            if init_node:
                init_sig = _extract_function_signature(init_node)
                class_sig = f"class {node.name}({init_sig.split('(', 1)[-1] if '(' in init_sig else ''})"
            else:
                class_sig = f"class {node.name}"

            signatures.append(UtilitySignature(
                module_path=module_path,
                name=node.name,
                kind="class",
                signature=class_sig,
                methods=methods,
            ))

    return signatures


def _extract_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract a human-readable function signature from an AST node."""
    args = []
    all_args = node.args

    # Positional args
    num_defaults = len(all_args.defaults)
    non_default_count = len(all_args.args) - num_defaults

    for i, arg in enumerate(all_args.args):
        if arg.arg == "self" or arg.arg == "cls":
            continue
        name = arg.arg
        annotation = ""
        if arg.annotation:
            annotation = f": {ast.unparse(arg.annotation)}"

        if i >= non_default_count:
            default_idx = i - non_default_count
            default_val = ast.unparse(all_args.defaults[default_idx])
            args.append(f"{name}{annotation} = {default_val}")
        else:
            args.append(f"{name}{annotation}")

    # *args
    if all_args.vararg:
        args.append(f"*{all_args.vararg.arg}")

    # **kwargs
    if all_args.kwarg:
        args.append(f"**{all_args.kwarg.arg}")

    # Return annotation
    ret = ""
    if node.returns:
        ret = f" -> {ast.unparse(node.returns)}"

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)}){ret}"


def _path_to_module(filepath: Path) -> str:
    """Convert a file path to a dotted module path.

    e.g., src/utils/logger.py -> src.utils.logger
    """
    parts = filepath.with_suffix("").parts
    # Strip leading drive letter / root on Windows
    clean_parts = []
    for p in parts:
        if ":" in p or p in ("/", "\\"):
            continue
        clean_parts.append(p)

    # Try to find 'src' as the root anchor
    if "src" in clean_parts:
        idx = clean_parts.index("src")
        clean_parts = clean_parts[idx:]

    return ".".join(clean_parts)


# ---------------------------------------------------------------------------
# Java: Regex-based crawling
# ---------------------------------------------------------------------------

# Matches: public/protected/private [static] [abstract] ReturnType methodName(params)
_JAVA_METHOD_RE = re.compile(
    r"^\s*(?:public|protected|private)\s+"
    r"(?:static\s+)?(?:abstract\s+)?(?:final\s+)?"
    r"(\w[\w<>\[\],\s]*?)\s+(\w+)\s*\(([^)]*)\)",
    re.MULTILINE,
)

# Matches: public/protected class ClassName [extends ...] [implements ...]
_JAVA_CLASS_RE = re.compile(
    r"^\s*(?:public|protected|private)?\s*(?:abstract\s+)?(?:final\s+)?"
    r"class\s+(\w+)",
    re.MULTILINE,
)


def _crawl_java(filepath: Path) -> list[UtilitySignature]:
    """Extract class and method signatures from a Java file using regex."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    module_path = str(filepath.with_suffix("")).replace("\\", ".").replace("/", ".")

    signatures = []

    # Find classes
    for match in _JAVA_CLASS_RE.finditer(source):
        class_name = match.group(1)

        # Find methods within this class
        methods = [m.group(2) for m in _JAVA_METHOD_RE.finditer(source)]

        signatures.append(UtilitySignature(
            module_path=module_path,
            name=class_name,
            kind="class",
            signature=f"class {class_name}",
            methods=methods,
        ))

    return signatures


# ---------------------------------------------------------------------------
# C#: Regex-based crawling
# ---------------------------------------------------------------------------

_CSHARP_METHOD_RE = re.compile(
    r"^\s*(?:public|protected|private|internal)\s+"
    r"(?:static\s+)?(?:async\s+)?(?:virtual\s+)?(?:override\s+)?"
    r"(\w[\w<>\[\],\s]*?)\s+(\w+)\s*\(([^)]*)\)",
    re.MULTILINE,
)

_CSHARP_CLASS_RE = re.compile(
    r"^\s*(?:public|protected|private|internal)?\s*(?:abstract\s+)?(?:sealed\s+)?"
    r"(?:partial\s+)?class\s+(\w+)",
    re.MULTILINE,
)


def _crawl_csharp(filepath: Path) -> list[UtilitySignature]:
    """Extract class and method signatures from a C# file using regex."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    module_path = str(filepath.with_suffix("")).replace("\\", ".").replace("/", ".")

    signatures = []

    for match in _CSHARP_CLASS_RE.finditer(source):
        class_name = match.group(1)
        methods = [m.group(2) for m in _CSHARP_METHOD_RE.finditer(source)]

        signatures.append(UtilitySignature(
            module_path=module_path,
            name=class_name,
            kind="class",
            signature=f"class {class_name}",
            methods=methods,
        ))

    return signatures
