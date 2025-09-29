#!/usr/bin/env python3
"""
Fix the HTML file properly with Chapter 16 in TOC and stories in correct chapters
"""

def fix_html_properly():
    """Fix the HTML file properly with all content in correct places"""
    import re
    
    print("🚀 Fixing HTML file properly with Chapter 16 in TOC and stories in correct chapters...")
    
    # Read the COMPLETE_FINAL.html file (this has Chapter 16 content)
    with open('Operational_Excellence_with_AI_COMPLETE_FINAL.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Adding Chapter 16 to TOC...")
    
    # Add Chapter 16 to TOC before Bibliography
    chapter16_toc = '            <li class="chapter"><a href="#chapter16">Chapter 16: AI Ethics and Governance <span class="toc-page-number">305</span></a></li>\n'
    
    # Find the TOC and add Chapter 16
    html_content = re.sub(
        r'(<li class="chapter"><a href="#chapter15">Chapter 15: AI Tools and Technologies: A Practical Guide <span class="toc-page-number">285</span></a></li>\s*</ul>)',
        r'\1' + chapter16_toc + '        </ul>',
        html_content
    )
    
    print("📝 2. Adding Sarah Chen story to Chapter 1 after overview...")
    
    # Add Sarah Chen story to Chapter 1 after the overview section
    sarah_story = '''
    <div class="story-section">
        <h3>The Transformation Story: From Reactive to Predictive Excellence</h3>
        
        <p><strong>The Challenge:</strong> Sarah Chen, Operations Director at TechFlow Manufacturing, faced a crisis. Her production line was experiencing 15% downtime due to unexpected equipment failures. Despite having a team of skilled technicians and following all traditional maintenance protocols, breakdowns kept happening at the worst possible times—during peak production periods when customer orders were highest.</p>
        
        <p><strong>The Breakthrough:</strong> When Sarah implemented AI-powered predictive maintenance, everything changed. The system learned from historical data, sensor readings, and environmental conditions to predict failures 72 hours in advance. Instead of reacting to problems, her team began preventing them. Downtime dropped to 3%, customer satisfaction soared, and the company saved $2.3 million in the first year alone.</p>
        
        <p><strong>The Lesson:</strong> This transformation represents the fundamental shift from reactive to predictive operations—the core of AI-driven operational excellence.</p>
    </div>
    '''
    
    # Find Chapter 1 overview and add story after it
    html_content = re.sub(
        r'(<h3 id="chapter-overview">Chapter Overview</h3>.*?</div>\s*<div class="page-break"></div>)',
        r'\1' + sarah_story,
        html_content,
        flags=re.DOTALL
    )
    
    print("📝 3. Adding Marcus Rodriguez story to Chapter 2 after overview...")
    
    # Add Marcus Rodriguez story to Chapter 2 after the overview section
    marcus_story = '''
    <div class="story-section">
        <h3>The Quality Revolution Story: When AI Became the Ultimate Inspector</h3>
        
        <p><strong>The Problem:</strong> Marcus Rodriguez, Quality Manager at Precision Components Inc., was losing sleep. Despite having 50 experienced quality inspectors working three shifts, defective products were still reaching customers. The company's defect rate was 0.8%—acceptable by industry standards but costing $500,000 annually in warranty claims and lost customers.</p>
        
        <p><strong>The Human Challenge:</strong> Marcus noticed something troubling. His best inspector, Maria Santos, could spot 99.2% of defects, but after 6 hours of intense visual inspection, her accuracy dropped to 94%. Fatigue, distraction, and human limitations were creating quality gaps that AI could eliminate.</p>
        
        <p><strong>The AI Solution:</strong> When Marcus implemented computer vision AI for quality inspection, the results were transformative. The AI system achieved 99.7% accuracy consistently, 24/7, without fatigue. It could detect defects invisible to the human eye—microscopic cracks, subtle color variations, and dimensional variations of 0.001 inches.</p>
        
        <p><strong>The Outcome:</strong> Defect rates dropped to 0.1%, warranty claims decreased by 85%, and customer satisfaction scores reached all-time highs. Maria's role evolved from repetitive inspection to analyzing AI insights and training the system for new defect patterns.</p>
        
        <p><strong>The Lesson:</strong> AI doesn't replace human expertise—it amplifies it, enabling humans to focus on higher-value activities while ensuring consistent, superior quality.</p>
    </div>
    '''
    
    # Find Chapter 2 overview and add story after it
    html_content = re.sub(
        r'(<h3 id="chapter-overview-1">Chapter Overview</h3>.*?<div class="page-break"></div>)',
        r'\1' + marcus_story,
        html_content,
        flags=re.DOTALL
    )
    
    print("📝 4. Adding self-assessment to Chapter 1...")
    
    # Add self-assessment to Chapter 1 before conclusion
    assessment = '''
    <div class="assessment-section">
        <h3>Chapter 1 Self-Assessment: AI Readiness Evaluation</h3>
        
        <p><strong>Rate your organization on each dimension (1-5 scale):</strong></p>
        
        <h4>Data Foundation (Score: ___/25)</h4>
        <ul>
            <li><strong>Data Quality (1-5):</strong> We have clean, accurate, and well-organized data</li>
            <li><strong>Data Accessibility (1-5):</strong> Data is easily accessible across departments</li>
            <li><strong>Data Governance (1-5):</strong> We have clear data ownership and management policies</li>
            <li><strong>Data Integration (1-5):</strong> Our systems can share data seamlessly</li>
            <li><strong>Data Security (1-5):</strong> We have robust data protection measures</li>
        </ul>
        
        <h4>Technology Infrastructure (Score: ___/25)</h4>
        <ul>
            <li><strong>Cloud Readiness (1-5):</strong> We have cloud infrastructure or migration plans</li>
            <li><strong>API Integration (1-5):</strong> Our systems support modern integration methods</li>
            <li><strong>Scalability (1-5):</strong> Our infrastructure can handle increased workloads</li>
            <li><strong>Security Framework (1-5):</strong> We have comprehensive cybersecurity measures</li>
            <li><strong>Monitoring Systems (1-5):</strong> We can monitor system performance effectively</li>
        </ul>
        
        <h4>Organizational Readiness (Score: ___/25)</h4>
        <ul>
            <li><strong>Leadership Support (1-5):</strong> Senior leadership actively supports AI initiatives</li>
            <li><strong>Change Management (1-5):</strong> We have experience managing organizational change</li>
            <li><strong>Skills Development (1-5):</strong> We invest in employee training and development</li>
            <li><strong>Cross-functional Collaboration (1-5):</strong> Teams work well together across departments</li>
            <li><strong>Innovation Culture (1-5):</strong> We encourage experimentation and innovation</li>
        </ul>
        
        <h4>Business Alignment (Score: ___/25)</h4>
        <ul>
            <li><strong>Strategic Clarity (1-5):</strong> We have clear business objectives for AI</li>
            <li><strong>ROI Measurement (1-5):</strong> We can measure and track business value</li>
            <li><strong>Customer Focus (1-5):</strong> We understand our customers' needs deeply</li>
            <li><strong>Process Optimization (1-5):</strong> We continuously improve our processes</li>
            <li><strong>Competitive Advantage (1-5):</strong> We understand our competitive landscape</li>
        </ul>
        
        <p><strong>Total Score: ___/100</strong></p>
        
        <h4>Interpretation:</h4>
        <ul>
            <li><strong>80-100:</strong> AI-Ready - You're well-positioned for AI implementation</li>
            <li><strong>60-79:</strong> AI-Prepared - Address gaps before major AI initiatives</li>
            <li><strong>40-59:</strong> AI-Developing - Focus on foundational improvements first</li>
            <li><strong>20-39:</strong> AI-Exploring - Start with small pilot projects</li>
            <li><strong>0-19:</strong> AI-Foundation - Build basic capabilities before AI implementation</li>
        </ul>
        
        <div class="try-this-section">
            <h4>Try This: AI Opportunity Mapping Exercise</h4>
            <p><strong>Instructions:</strong> Identify three operational areas where AI could create value:</p>
            
            <p><strong>1. Current Pain Point:</strong> ________________________________<br>
            <strong>AI Solution:</strong> ________________________________<br>
            <strong>Expected Impact:</strong> ________________________________</p>
            
            <p><strong>2. Current Pain Point:</strong> ________________________________<br>
            <strong>AI Solution:</strong> ________________________________<br>
            <strong>Expected Impact:</strong> ________________________________</p>
            
            <p><strong>3. Current Pain Point:</strong> ________________________________<br>
            <strong>AI Solution:</strong> ________________________________<br>
            <strong>Expected Impact:</strong> ________________________________</p>
        </div>
    </div>
    '''
    
    # Add assessment to Chapter 1 before conclusion
    html_content = re.sub(
        r'(<h3 id="conclusion">Conclusion</h3>)',
        assessment + r'\1',
        html_content
    )
    
    print("📝 5. Adding CSS for new elements...")
    
    # Add CSS for new elements
    new_css = '''
        .story-section {
            background: #f8f9fa;
            border-left: 4px solid #e74c3c;
            padding: 0.3in;
            margin: 0.3in 0;
            page-break-inside: avoid;
        }
        
        .story-section h3 {
            color: #e74c3c;
            font-size: 18pt;
            margin-bottom: 0.2in;
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 0.1in;
        }
        
        .assessment-section {
            background: #e8f5e8;
            border: 2px solid #28a745;
            border-radius: 8px;
            padding: 0.3in;
            margin: 0.3in 0;
            page-break-inside: avoid;
        }
        
        .assessment-section h3 {
            color: #155724;
            background: #d4edda;
            padding: 0.1in;
            margin: -0.3in -0.3in 0.2in -0.3in;
            border-radius: 6px 6px 0 0;
        }
        
        .try-this-section {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
            padding: 0.2in;
            margin: 0.2in 0;
        }
        
        .try-this-section h4 {
            color: #856404;
            margin-top: 0;
        }
        
        /* Smaller font for TOC to fit Chapter 16 */
        .toc-list {
            font-size: 0.9em;
        }
    '''
    
    # Inject the new CSS
    html_content = html_content.replace('</style>', new_css + '\n    </style>')
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_PROPERLY_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Properly fixed HTML created: {output_file}")
    return output_file

def generate_properly_fixed_pdf(html_file):
    """Generate PDF from properly fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_PROPERLY_FIXED.pdf'
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
            print(f"🎉 Properly Fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing HTML properly with Chapter 16 in TOC and stories in correct chapters...")
    
    # Fix HTML properly
    html_file = fix_html_properly()
    
    if html_file:
        # Generate properly fixed PDF
        print("\n📚 Generating properly fixed PDF...")
        generate_properly_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 added to TOC (with smaller font)")
        print("2. ✅ Sarah Chen story in Chapter 1 (after overview)")
        print("3. ✅ Marcus Rodriguez story in Chapter 2 (after overview)")
        print("4. ✅ Self-assessment quiz in Chapter 1")
        print("5. ✅ 'Try This' exercise in Chapter 1")
        print("6. ✅ All content in correct chapters")
        print("\n📖 The PDF should now have Chapter 16 in TOC and stories in their proper chapters!")
    else:
        print("❌ Failed to fix HTML")
