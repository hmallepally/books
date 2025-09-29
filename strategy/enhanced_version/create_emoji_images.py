#!/usr/bin/env python3
"""
Create 16x16 PNG images to replace emojis in the Living Strategy book.
These images will be print-friendly and KDP-compatible.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_rocket_image():
    """Create a rocket icon (🚀)"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Rocket body (vertical rectangle)
    draw.rectangle([6, 2, 10, 12], fill='black')
    
    # Rocket nose (triangle)
    draw.polygon([(8, 0), (5, 3), (11, 3)], fill='black')
    
    # Rocket fins (bottom triangles)
    draw.polygon([(6, 12), (4, 16), (6, 14)], fill='black')
    draw.polygon([(10, 12), (12, 16), (10, 14)], fill='black')
    
    # Rocket flame (small lines)
    draw.line([(7, 14), (7, 16)], fill='black', width=1)
    draw.line([(9, 14), (9, 16)], fill='black', width=1)
    
    return img

def create_lightbulb_image():
    """Create a lightbulb icon (💡)"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Lightbulb body (circle)
    draw.ellipse([4, 2, 12, 10], fill='black')
    
    # Lightbulb base (rectangle)
    draw.rectangle([5, 10, 11, 12], fill='black')
    
    # Lightbulb screw base (smaller rectangle)
    draw.rectangle([6, 12, 10, 14], fill='black')
    
    # Light rays (small lines)
    draw.line([(2, 4), (4, 4)], fill='black', width=1)
    draw.line([(12, 4), (14, 4)], fill='black', width=1)
    draw.line([(2, 6), (4, 6)], fill='black', width=1)
    draw.line([(12, 6), (14, 6)], fill='black', width=1)
    
    return img

def create_tools_image():
    """Create a tools icon (🛠️)"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Hammer head (rectangle)
    draw.rectangle([2, 2, 6, 4], fill='black')
    
    # Hammer handle (line)
    draw.line([(4, 4), (4, 12)], fill='black', width=2)
    
    # Wrench (L-shape)
    draw.line([(10, 2), (14, 2)], fill='black', width=2)
    draw.line([(14, 2), (14, 6)], fill='black', width=2)
    draw.line([(10, 6), (14, 6)], fill='black', width=2)
    
    return img

def create_book_image():
    """Create a book icon (📖)"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Book cover (rectangle)
    draw.rectangle([3, 2, 13, 14], fill='black')
    
    # Book pages (smaller rectangle)
    draw.rectangle([4, 3, 12, 13], fill='white')
    
    # Book spine (vertical line)
    draw.line([(3, 2), (3, 14)], fill='black', width=1)
    
    # Book text lines
    draw.line([(5, 5), (11, 5)], fill='black', width=1)
    draw.line([(5, 7), (11, 7)], fill='black', width=1)
    draw.line([(5, 9), (11, 9)], fill='black', width=1)
    draw.line([(5, 11), (11, 11)], fill='black', width=1)
    
    return img

def create_chart_image():
    """Create a chart icon (📈)"""
    img = Image.new('RGBA', (16, 16), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Chart axes
    draw.line([(2, 14), (14, 14)], fill='black', width=1)  # X-axis
    draw.line([(2, 2), (2, 14)], fill='black', width=1)   # Y-axis
    
    # Chart line (ascending)
    draw.line([(3, 12), (6, 9), (9, 6), (12, 3)], fill='black', width=2)
    
    # Data points (small circles)
    draw.ellipse([(2, 11), (4, 13)], fill='black')
    draw.ellipse([(5, 8), (7, 10)], fill='black')
    draw.ellipse([(8, 5), (10, 7)], fill='black')
    draw.ellipse([(11, 2), (13, 4)], fill='black')
    
    return img

def main():
    """Create all emoji replacement images"""
    # Create images directory if it doesn't exist
    os.makedirs('strategy/enhanced_version/images', exist_ok=True)
    
    # Create each image
    images = {
        'rocket': create_rocket_image(),
        'lightbulb': create_lightbulb_image(),
        'tools': create_tools_image(),
        'book': create_book_image(),
        'chart': create_chart_image()
    }
    
    # Save images
    for name, img in images.items():
        filename = f'strategy/enhanced_version/images/{name}.png'
        img.save(filename)
        print(f"Created {filename}")
    
    print("\nAll emoji replacement images created successfully!")
    print("Files created:")
    print("- rocket.png (🚀) - for 'What Changed' sections")
    print("- lightbulb.png (💡) - for 'Key Principle' sections")
    print("- tools.png (🛠️) - for 'Practical Application' sections")
    print("- book.png (📖) - for 'Case Reflection' sections")
    print("- chart.png (📈) - for 'Action Steps' sections")

if __name__ == "__main__":
    main()

