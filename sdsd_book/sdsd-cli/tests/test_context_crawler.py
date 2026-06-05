"""Tests for the context crawler module."""

from pathlib import Path

from sdsd.core.context_crawler import crawl_target, detect_language


FIXTURES = Path(__file__).parent / "fixtures" / "sample_project"


class TestDetectLanguage:
    """Tests for language detection."""

    def test_detect_python(self):
        result = detect_language(FIXTURES / "src" / "payments")
        assert result == "python"

    def test_detect_default_for_empty(self, tmp_path):
        result = detect_language(tmp_path)
        assert result == "python"  # Default


class TestCrawlPython:
    """Tests for Python AST-based crawling."""

    def test_crawl_finds_classes(self):
        results = crawl_target(FIXTURES / "src" / "payments", "python")
        class_names = [r.name for r in results if r.kind == "class"]
        assert "StructuredLogger" in class_names

    def test_crawl_finds_functions(self):
        results = crawl_target(FIXTURES / "src" / "payments", "python")
        func_names = [r.name for r in results if r.kind == "function"]
        assert "calculate_fee" in func_names
        assert "process_payment" in func_names

    def test_crawl_extracts_methods(self):
        results = crawl_target(FIXTURES / "src" / "payments", "python")
        logger = [r for r in results if r.name == "StructuredLogger"][0]
        assert "info" in logger.methods
        assert "error" in logger.methods
        assert "audit" in logger.methods

    def test_crawl_extracts_async_signature(self):
        results = crawl_target(FIXTURES / "src" / "payments", "python")
        payment_fn = [r for r in results if r.name == "process_payment"][0]
        assert "async def" in payment_fn.signature

    def test_crawl_extracts_default_args(self):
        results = crawl_target(FIXTURES / "src" / "payments", "python")
        fee_fn = [r for r in results if r.name == "calculate_fee"][0]
        assert "0.025" in fee_fn.signature

    def test_crawl_nonexistent_path(self):
        results = crawl_target(Path("/nonexistent/path"))
        assert results == []

    def test_crawl_single_file(self):
        results = crawl_target(
            FIXTURES / "src" / "payments" / "processor.py", "python"
        )
        assert len(results) >= 3  # 1 class + 2 functions
