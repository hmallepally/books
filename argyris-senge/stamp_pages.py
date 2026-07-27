"""
Stamp page numbers onto a PDF file.
Adds centered page numbers at the bottom of each page,
starting from page 5 (skipping title, copyright, dedication, preface front matter).
Numbers are baked into the content layer so KDP preserves them.
"""

import sys
import os
from io import BytesIO
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


def create_page_number_overlay(width, height, page_num):
    """Create a single-page PDF with just a page number centered at the bottom."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=(width, height))
    c.setFont("Times-Roman", 9)
    c.setFillColorRGB(0.33, 0.33, 0.33)  # #555 equivalent
    # Center the page number at bottom of page, 0.4in from bottom edge
    c.drawCentredString(width / 2, 0.4 * inch, str(page_num))
    c.save()
    buffer.seek(0)
    return PdfReader(buffer).pages[0]


def stamp_pdf(pdf_path, start_page=5):
    """Add page numbers to a PDF, starting from start_page (1-indexed)."""
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    for i, page in enumerate(reader.pages):
        page_num = i + 1  # 1-indexed
        if page_num >= start_page:
            # Get page dimensions
            mediabox = page.mediabox
            width = float(mediabox.width)
            height = float(mediabox.height)
            # Create and merge the page number overlay
            overlay = create_page_number_overlay(width, height, page_num)
            page.merge_page(overlay)
        writer.add_page(page)

    # Write to a temp file first, then replace
    temp_path = pdf_path + '.tmp'
    with open(temp_path, 'wb') as f:
        writer.write(f)

    os.replace(temp_path, pdf_path)
    print(f"Stamped page numbers on pages {start_page}-{len(reader.pages)}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stamp_pages.py <pdf_path> [start_page]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    start_page = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    stamp_pdf(pdf_path, start_page)
