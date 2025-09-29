#!/usr/bin/env python3
"""
Simple emoji replacement image creator using basic drawing
"""

import os

def create_simple_images():
    """Create simple text-based images that can be converted to PNG"""
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    # Create simple SVG files for each emoji
    svg_files = {
        'rocket.svg': '''<svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
  <rect x="6" y="2" width="4" height="10" fill="black"/>
  <polygon points="8,0 5,3 11,3" fill="black"/>
  <polygon points="6,12 4,16 6,14" fill="black"/>
  <polygon points="10,12 12,16 10,14" fill="black"/>
  <line x1="7" y1="14" x2="7" y2="16" stroke="black" stroke-width="1"/>
  <line x1="9" y1="14" x2="9" y2="16" stroke="black" stroke-width="1"/>
</svg>''',
        
        'lightbulb.svg': '''<svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="8" cy="6" rx="4" ry="4" fill="black"/>
  <rect x="5" y="10" width="6" height="2" fill="black"/>
  <rect x="6" y="12" width="4" height="2" fill="black"/>
  <line x1="2" y1="4" x2="4" y2="4" stroke="black" stroke-width="1"/>
  <line x1="12" y1="4" x2="14" y2="4" stroke="black" stroke-width="1"/>
  <line x1="2" y1="6" x2="4" y2="6" stroke="black" stroke-width="1"/>
  <line x1="12" y1="6" x2="14" y2="6" stroke="black" stroke-width="1"/>
</svg>''',
        
        'tools.svg': '''<svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
  <rect x="2" y="2" width="4" height="2" fill="black"/>
  <line x1="4" y1="4" x2="4" y2="12" stroke="black" stroke-width="2"/>
  <line x1="10" y1="2" x2="14" y2="2" stroke="black" stroke-width="2"/>
  <line x1="14" y1="2" x2="14" y2="6" stroke="black" stroke-width="2"/>
  <line x1="10" y1="6" x2="14" y2="6" stroke="black" stroke-width="2"/>
</svg>''',
        
        'book.svg': '''<svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
  <rect x="3" y="2" width="10" height="12" fill="black"/>
  <rect x="4" y="3" width="8" height="10" fill="white"/>
  <line x1="3" y1="2" x2="3" y2="14" stroke="black" stroke-width="1"/>
  <line x1="5" y1="5" x2="11" y2="5" stroke="black" stroke-width="1"/>
  <line x1="5" y1="7" x2="11" y2="7" stroke="black" stroke-width="1"/>
  <line x1="5" y1="9" x2="11" y2="9" stroke="black" stroke-width="1"/>
  <line x1="5" y1="11" x2="11" y2="11" stroke="black" stroke-width="1"/>
</svg>''',
        
        'chart.svg': '''<svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
  <line x1="2" y1="14" x2="14" y2="14" stroke="black" stroke-width="1"/>
  <line x1="2" y1="2" x2="2" y2="14" stroke="black" stroke-width="1"/>
  <line x1="3" y1="12" x2="6" y2="9" stroke="black" stroke-width="2"/>
  <line x1="6" y1="9" x2="9" y2="6" stroke="black" stroke-width="2"/>
  <line x1="9" y1="6" x2="12" y2="3" stroke="black" stroke-width="2"/>
  <circle cx="3" cy="12" r="1" fill="black"/>
  <circle cx="6" cy="9" r="1" fill="black"/>
  <circle cx="9" cy="6" r="1" fill="black"/>
  <circle cx="12" cy="3" r="1" fill="black"/>
</svg>'''
    }
    
    # Write SVG files
    for filename, content in svg_files.items():
        with open(f'images/{filename}', 'w') as f:
            f.write(content)
        print(f"Created images/{filename}")
    
    print("\nSVG files created successfully!")
    print("You can convert these to PNG using online tools or image editors.")
    print("Files created:")
    print("- rocket.svg (🚀) - for 'What Changed' sections")
    print("- lightbulb.svg (💡) - for 'Key Principle' sections") 
    print("- tools.svg (🛠️) - for 'Practical Application' sections")
    print("- book.svg (📖) - for 'Case Reflection' sections")
    print("- chart.svg (📈) - for 'Action Steps' sections")

if __name__ == "__main__":
    create_simple_images()

