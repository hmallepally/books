#!/usr/bin/env python3
"""
Create Separate Components for 10-20-70 Framework
"""

def create_separate_components():
    """Create separate Mermaid diagrams for each component"""
    
    print("🎨 Creating separate components for 10-20-70 framework...")
    
    # Create the 70% People & Process component
    people_process_mmd = """
    graph TD
        A["70% People & Process"] --> B["Foster AI-Driven Culture"]
        A --> C["Upskill Workforce"]
        A --> D["Cross-Functional Teams"]
        
        B --> E["Leadership commitment<br/>Change management<br/>Continuous learning"]
        C --> F["Technical training<br/>Data literacy<br/>AI awareness"]
        D --> G["IT + Business<br/>Data scientists + Operations<br/>Collaborative approach"]
        
        style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
        style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
        style C fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
        style D fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    """
    
    # Create the 20% Data & Infrastructure component
    data_infra_mmd = """
    graph TD
        A["20% Data & Infrastructure"] --> B["Robust Data Pipelines"]
        A --> C["Data Quality Management"]
        A --> D["Scalable Infrastructure"]
        
        B --> E["Data collection<br/>ETL processes<br/>Real-time streaming"]
        C --> F["Data validation<br/>Cleansing<br/>Governance"]
        D --> G["Cloud platforms<br/>MLOps<br/>Security & Privacy"]
        
        style A fill:#e8f5e8,stroke:#1b5e20,stroke-width:3px
        style B fill:#fff3e0,stroke:#e65100,stroke-width:2px
        style C fill:#fff3e0,stroke:#e65100,stroke-width:2px
        style D fill:#fff3e0,stroke:#e65100,stroke-width:2px
    """
    
    # Create the 10% Algorithms & Models component
    algorithms_mmd = """
    graph TD
        A["10% Algorithms & Models"] --> B["Cutting-Edge ML"]
        A --> C["Custom Models"]
        A --> D["Ethical AI"]
        
        B --> E["Deep Learning<br/>NLP<br/>Computer Vision"]
        C --> F["Custom development<br/>Open-source integration<br/>Model optimization"]
        D --> G["Interpretability<br/>Bias detection<br/>Fairness"]
        
        style A fill:#fce4ec,stroke:#880e4f,stroke-width:3px
        style B fill:#f1f8e9,stroke:#33691e,stroke-width:2px
        style C fill:#f1f8e9,stroke:#33691e,stroke-width:2px
        style D fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    """
    
    # Save each component
    components = [
        ("people_process", people_process_mmd),
        ("data_infrastructure", data_infra_mmd),
        ("algorithms_models", algorithms_mmd)
    ]
    
    for name, content in components:
        filename = f"diagrams/{name}_component.mmd"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created {filename}")
    
    return components

def generate_component_images():
    """Generate PNG images from Mermaid files"""
    import subprocess
    import os
    
    print("\n🖼️ Generating component images...")
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    components = ['people_process', 'data_infrastructure', 'algorithms_models']
    
    for component in components:
        input_file = f"diagrams/{component}_component.mmd"
        output_file = f"images/{component}_component.png"
        
        try:
            # Generate high-resolution image
            subprocess.run([
                'mmdc', 
                '-i', input_file,
                '-o', output_file,
                '-w', '1200',
                '-H', '800',
                '-b', 'white',
                '-s', '2'
            ], check=True)
            print(f"✅ Generated {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate {output_file}: {e}")
        except FileNotFoundError:
            print(f"❌ mmdc not found. Please install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")

def update_html_with_components():
    """Update HTML to use separate components instead of single image"""
    import re
    
    print("\n📝 Updating HTML with separate components...")
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_IMAGE_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Replace the single image with three separate components
    new_content = '''
    <p>The AI Success Framework consists of three critical components, each requiring different levels of investment and focus:</p>
    
    <h4>70% People & Process</h4>
    <p>The foundation of AI success lies in organizational culture and human capabilities. This includes fostering an AI-driven mindset, upskilling the workforce, and creating cross-functional teams that can effectively leverage AI technologies.</p>
    <figure>
    <img src="images/people_process_component.png" alt="70% People & Process Component" style="max-width: 100%; margin: 0.3in auto; display: block;" />
    <figcaption aria-hidden="true">70% People & Process Component</figcaption>
    </figure>
    
    <h4>20% Data & Infrastructure</h4>
    <p>Robust data pipelines and scalable infrastructure form the technical backbone of AI implementations. This includes data quality management, governance frameworks, and cloud-based platforms that can support AI workloads.</p>
    <figure>
    <img src="images/data_infrastructure_component.png" alt="20% Data & Infrastructure Component" style="max-width: 100%; margin: 0.3in auto; display: block;" />
    <figcaption aria-hidden="true">20% Data & Infrastructure Component</figcaption>
    </figure>
    
    <h4>10% Algorithms & Models</h4>
    <p>While algorithms and models are often the most visible aspect of AI, they represent only a small portion of the overall success. This includes cutting-edge machine learning techniques, custom model development, and ethical AI practices.</p>
    <figure>
    <img src="images/algorithms_models_component.png" alt="10% Algorithms & Models Component" style="max-width: 100%; margin: 0.3in auto; display: block;" />
    <figcaption aria-hidden="true">10% Algorithms & Models Component</figcaption>
    </figure>
    '''
    
    # Replace the old single image section
    html_content = re.sub(
        r'<p>The framework below illustrates how successful AI implementations balance three critical components:</p>\s*<figure>.*?</figure>',
        new_content,
        html_content,
        flags=re.DOTALL
    )
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_COMPONENTS.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Updated HTML created: {output_file}")
    return output_file

def generate_components_pdf(html_file):
    """Generate PDF from components HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_COMPONENTS.pdf'
            await page.pdf(
                path=output_pdf,
                format='A4',
                margin={
                    'top': '0.75in',
                    'right': '0.75in',
                    'bottom': '0.75in',
                    'left': '0.75in'
                },
                print_background=True,
                prefer_css_page_size=True,
                display_header_footer=False,
                tagged=True
            )
            
            await browser.close()
            print(f"🎉 Components PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    import os
    
    print("🚀 Creating separate components for 10-20-70 framework...")
    
    # Create diagrams directory if it doesn't exist
    os.makedirs('diagrams', exist_ok=True)
    
    # Create separate components
    create_separate_components()
    
    # Generate images
    generate_component_images()
    
    # Update HTML
    html_file = update_html_with_components()
    
    if html_file:
        # Generate PDF
        print("\n📚 Generating components PDF...")
        generate_components_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Separate components created:")
        print("1. ✅ 70% People & Process component")
        print("2. ✅ 20% Data & Infrastructure component") 
        print("3. ✅ 10% Algorithms & Models component")
        print("4. ✅ Each component in separate rows for readability")
        print("5. ✅ Added explanatory text for each component")
        print("\n📖 Please review this PDF!")
        print("🔍 Check if the separate components are now much more readable.")
    else:
        print("❌ Failed to create separate components")
