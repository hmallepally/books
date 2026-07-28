"""
Stamp page numbers onto a PDF file.
Adds centered page numbers at the bottom of each page,
starting from a configurable start page (skipping front matter).

Optimized: injects raw PDF content stream operations directly
instead of creating per-page overlay PDFs via ReportLab,
eliminating font re-embedding bloat entirely.
"""

import sys
import os
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, DictionaryObject, NameObject,
    NumberObject, DecodedStreamObject
)


def _ensure_times_roman(writer):
    """Add a Times-Roman Type1 font resource to the writer and return its ref.
    Type1 base-14 fonts don't need embedding — PDF readers have them built in."""
    font_dict = DictionaryObject()
    font_dict[NameObject("/Type")] = NameObject("/Font")
    font_dict[NameObject("/Subtype")] = NameObject("/Type1")
    font_dict[NameObject("/BaseFont")] = NameObject("/Times-Roman")
    return writer._add_object(font_dict)


def stamp_pdf(pdf_path, start_page=5, output_path=None):
    """Add page numbers with minimal size overhead using raw content streams."""
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    # Clone all pages into writer first
    writer.append(reader)

    # Create ONE shared font object
    font_ref = _ensure_times_roman(writer)

    for i, page in enumerate(writer.pages):
        page_num = i + 1
        if page_num < start_page:
            continue

        content_page_num = page_num - start_page + 1
        mediabox = page.mediabox
        width = float(mediabox.width)
        y_pos = 28.8  # 0.4 inch from bottom

        # Calculate text width to center it (approximate for Times-Roman 9pt)
        text = str(content_page_num)
        char_width = 4.5  # avg width per digit in Times-Roman 9pt
        text_width = len(text) * char_width
        x_pos = (width - text_width) / 2

        # Raw PDF content stream: set font, color, position, draw text
        stream_content = (
            f"q\n"                          # save graphics state
            f"/PageNum 9 Tf\n"              # set font
            f"0.33 0.33 0.33 rg\n"          # set fill color (gray)
            f"BT\n"                         # begin text
            f"{x_pos:.2f} {y_pos:.2f} Td\n" # position
            f"({text}) Tj\n"                # draw text
            f"ET\n"                         # end text
            f"Q\n"                          # restore graphics state
        )

        # Create the content stream object
        stream = DecodedStreamObject()
        stream.set_data(stream_content.encode('latin-1'))

        stream_ref = writer._add_object(stream)

        # Add font reference to page resources
        # Resolve indirect objects to get mutable dictionaries
        from pypdf.generic import IndirectObject
        resources = page.get("/Resources")
        if resources is None:
            resources = DictionaryObject()
        elif isinstance(resources, IndirectObject):
            resources = resources.get_object()
        
        fonts = resources.get("/Font")
        if fonts is None:
            fonts = DictionaryObject()
        elif isinstance(fonts, IndirectObject):
            fonts = DictionaryObject(fonts.get_object())

        fonts[NameObject("/PageNum")] = font_ref
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources

        # Append our content stream to existing page contents
        existing_contents = page.get("/Contents")
        if existing_contents is not None:
            if isinstance(existing_contents, ArrayObject):
                existing_contents.append(stream_ref)
            else:
                page[NameObject("/Contents")] = ArrayObject(
                    [existing_contents, stream_ref]
                )
        else:
            page[NameObject("/Contents")] = stream_ref

    # Write output
    if output_path is None:
        temp_path = pdf_path + '.tmp'
        with open(temp_path, 'wb') as f:
            writer.write(f)
        os.replace(temp_path, pdf_path)
        print(f"Stamped page numbers on pages {start_page}-{len(reader.pages)} (in-place)")
    else:
        with open(output_path, 'wb') as f:
            writer.write(f)
        print(f"Stamped page numbers on pages {start_page}-{len(reader.pages)} -> {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stamp_pages.py <pdf_path> [start_page] [output_path]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    start_page = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    stamp_pdf(pdf_path, start_page, output_path)
