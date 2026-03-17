Manmade
=======

This folder contains source material and a small build script to create an EPUB for the "Manmade" book — a collection of human inventions (good and bad) presented as a simple Kindle-ready EPUB.

What's included
- `content/` - markdown chapters and front/back matter
- `images/` - cover and per-chapter images (placeholders)
- `build_epub.py` - minimal Python script to convert markdown to EPUB using Pandoc (or markdown->EPUB via markdown2 + ebooklib fallback)
- `requirements.txt` - python packages used by `build_epub.py`

How to build (Windows PowerShell):

Install the Python dependencies and run the build script. If you have Pandoc installed, the script will prefer Pandoc for higher-quality output.

1. Install dependencies:

    pip install -r requirements.txt

2. Run the build:

    python build_epub.py

The built EPUB will be written to `dist/manmade.epub`.

Notes
- This is a scaffold. Add or edit files in `content/` to change book text. Replace images in `images/` with your artwork.
