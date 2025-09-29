#!/usr/bin/env python3

import pdfkit
import os

def generate_print_pdf():
    # Configure options for print size 8.5 x 11 inches
    options = {
        'page-size': 'Letter',  # 8.5 x 11 inches
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None,
        'print-media-type': None,
        'disable-smart-shrinking': None,
        'zoom': '1.0',
        'dpi': '300',
        'image-quality': '94',
        'image-dpi': '300',
        'load-error-handling': 'ignore',
        'load-media-error-handling': 'ignore'
    }

    try:
        # Configure wkhtmltopdf path
        config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
        
        # Generate PDF
        pdfkit.from_file(
            'Operational_Excellence_with_AI_WITH_TOC.html',
            'Operational_Excellence_with_AI_Print_Ready.pdf',
            options=options,
            configuration=config
        )
        
        print('PDF generated successfully: Operational_Excellence_with_AI_Print_Ready.pdf')
        print('PDF is optimized for 8.5 x 11 inch print size (Letter)')
        print(f'File size: {os.path.getsize("Operational_Excellence_with_AI_Print_Ready.pdf") / 1024 / 1024:.2f} MB')
        return True
        
    except Exception as e:
        print(f'Error generating PDF: {e}')
        print('Make sure wkhtmltopdf is installed on your system')
        return False

if __name__ == '__main__':
    generate_print_pdf()
