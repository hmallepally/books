"""Build a simple EPUB for the Manmade book from markdown files.

This script looks for markdown files in the `content/` directory, converts them to HTML,
and writes a minimal EPUB to `dist/manmade.epub`.

It prefers Pandoc (if installed) for conversion, otherwise uses markdown2 + EbookLib.
"""
import os
import sys
from pathlib import Path
import re
import mimetypes

try:
    import markdown2
except Exception:
    markdown2 = None

try:
    from ebooklib import epub
except Exception:
    print("Please install EbookLib (pip install EbookLib)")
    raise


ROOT = Path(__file__).parent
CONTENT_DIR = ROOT / "content"
IMAGES_DIR = ROOT / "images"
DIST_DIR = ROOT / "dist"
DIST_DIR.mkdir(exist_ok=True)


def md_to_html(md_text: str) -> str:
    # Prefer pandoc if available
    try:
        import pypandoc

        return pypandoc.convert_text(md_text, "html", format="md")
    except Exception:
        if markdown2:
            return markdown2.markdown(md_text)
        else:
            raise RuntimeError("No markdown converter available. Install pypandoc or markdown2.")


def build_epub(out_path: Path):
    book = epub.EpubBook()
    book.set_identifier("manmade-001")
    book.set_title("Manmade")
    book.set_language("en")
    # author will be set below

    # collect chapters
    files = sorted(CONTENT_DIR.glob("*.md"))

    spine = ["nav"]
    toc = []

    added_images = set()
    for i, f in enumerate(files, start=1):
        text = f.read_text(encoding="utf-8")

        # normalize our simple Image: lines to markdown image syntax so converter handles them
        # e.g. "Image: ../images/wheel.png" -> "![](../images/wheel.png)"
        text = re.sub(r"^Image:\s*(.+)$", r"![](\1)", text, flags=re.M)

        html = md_to_html(text)

        # find image src attributes in produced HTML and embed those files into the EPUB
        img_srcs = set(re.findall(r'src=["\']([^"\']+)["\']', html))
        for src in img_srcs:
            # resolve relative paths from the markdown file location
            if src.startswith('http://') or src.startswith('https://'):
                continue
            # normalize path (content files live in content/, images in ../images/)
            src_path = (f.parent / src).resolve()
            if not src_path.exists():
                # try swapping extension to svg if png not found
                alt = Path(str(src_path.with_suffix('.svg')))
                if alt.exists():
                    src_path = alt
                else:
                    # also try images/ relative to project root
                    root_alt = (Path(__file__).parent / src).resolve()
                    if root_alt.exists():
                        src_path = root_alt
            if src_path.exists():
                media_type = mimetypes.guess_type(src_path.name)[0] or 'image/png'
                epub_path = f'images/{src_path.name}'
                if src_path.name not in added_images:
                    with open(src_path, 'rb') as imgf:
                        item = epub.EpubItem(uid=src_path.name, file_name=epub_path, media_type=media_type, content=imgf.read())
                        book.add_item(item)
                    added_images.add(src_path.name)
                # replace source reference in HTML to the embedded path
                html = html.replace(src, epub_path)

        chap = epub.EpubHtml(title=f.stem, file_name=f"chap_{i}.xhtml", lang="en")
        chap.content = html.encode("utf-8")
        book.add_item(chap)
        toc.append(chap)
        spine.append(chap)

    # set author metadata if provided via environment or variable
    # default to empty string if not set
    AUTHOR = "Hari Mallepally"
    if AUTHOR:
        book.add_author(AUTHOR)

    # cover if present (try png then svg)
    # ensure a PNG cover exists by rendering cover.svg -> cover.png using cairosvg when available
    cover_path = IMAGES_DIR / "cover.png"
    if not cover_path.exists():
        svg = IMAGES_DIR / "cover.svg"
        if svg.exists():
            try:
                import cairosvg

                # render to the desired 1600x2560 size
                out_cover = IMAGES_DIR / "cover.png"
                cairosvg.svg2png(url=str(svg), write_to=str(out_cover), output_width=1600, output_height=2560)
                cover_path = out_cover
            except Exception:
                cover_path = svg
    if cover_path.exists():
        try:
            book.set_cover(cover_path.name, cover_path.read_bytes())
        except Exception:
            # fallback: ignore cover if it fails
            pass

    # include CSS if present
    css_path = ROOT / "epub-styles.css"
    if css_path.exists():
        style_item = epub.EpubItem(uid="style_nav", file_name="styles/epub-styles.css", media_type="text/css", content=css_path.read_bytes())
        book.add_item(style_item)
        # attach CSS to each chapter
        for item in list(book.get_items_of_type(epub.EpubHtml)):
            item.add_link(href="styles/epub-styles.css", rel="stylesheet", type="text/css")

    # basic nav
    book.toc = tuple(toc)
    book.spine = spine

    # add default NCX and Nav files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # write
    epub.write_epub(str(out_path), book)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out = DIST_DIR / "manmade.epub"
    build_epub(out)
