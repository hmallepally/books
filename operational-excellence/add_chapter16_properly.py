#!/usr/bin/env python3
"""
Properly add Chapter 16 to the HTML file
"""

def properly_add_chapter16():
    """Add Chapter 16 properly to the HTML file"""
    import re
    
    print("🚀 Properly adding Chapter 16 to HTML file...")
    
    # Read the HTML file
    with open('Operational_Excellence_with_AI_COMPLETE_ENHANCED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 Adding Chapter 16: AI Ethics & Governance...")
    
    # Chapter 16 content
    chapter16_content = '''
<div class="page-break"></div>

<div id="chapter16" class="chapter">
    <h1 class="chapter-title">Chapter 16: AI Ethics and Governance: Building Responsible AI Systems</h1>
    
    <div class="quote-chapter">
        <blockquote>
            <strong>"The development of full artificial intelligence could spell the end of the human race. But it could also be the beginning of a new era of human flourishing—if we get the governance right."</strong><br>
            – Adapted from Stephen Hawking
        </blockquote>
    </div>
    
    <h3>Chapter Overview</h3>
    
    <p>This chapter addresses one of the most critical aspects of AI implementation: ensuring that AI systems are developed and deployed responsibly. We explore ethical considerations, governance frameworks, and practical approaches to building AI systems that are fair, transparent, and aligned with human values.</p>
    
    <div class="page-break"></div>
    
    <div class="story-section">
        <h3>The Ethics Imperative: When AI Decisions Affect Lives</h3>
        
        <p><strong>The Challenge:</strong> Dr. Sarah Kim, Chief AI Officer at MedTech Solutions, faced a moral dilemma. Her company's AI system for prioritizing patient care was showing bias against certain demographic groups. The algorithm was inadvertently deprioritizing patients from underserved communities, potentially affecting life-or-death medical decisions.</p>
        
        <p><strong>The Discovery:</strong> During routine auditing, Sarah's team found that the AI system was 23% less likely to recommend urgent care for patients from low-income neighborhoods. The bias wasn't intentional—it was embedded in the historical data used to train the system, which reflected existing healthcare disparities.</p>
        
        <p><strong>The Response:</strong> Sarah immediately halted the system's deployment and led a comprehensive review. Her team implemented fairness constraints, diversified the training data, and created ongoing monitoring systems. The revised system not only eliminated bias but actually improved outcomes for underserved populations.</p>
        
        <p><strong>The Lesson:</strong> AI ethics isn't an afterthought—it's a fundamental requirement that must be built into every AI system from the ground up.</p>
    </div>
    
    <h3>Understanding AI Ethics: The Foundation of Responsible AI</h3>
    
    <p>AI ethics encompasses the moral principles and values that guide the development, deployment, and use of artificial intelligence systems. As AI becomes more powerful and pervasive, ensuring ethical AI practices becomes not just a moral imperative but a business necessity.</p>
    
    <h4>The Four Pillars of AI Ethics</h4>
    
    <h5>1. Fairness and Non-Discrimination</h5>
    <ul>
        <li>Ensuring AI systems treat all individuals and groups equitably</li>
        <li>Preventing bias in training data and algorithms</li>
        <li>Regular auditing for discriminatory outcomes</li>
        <li>Diverse representation in AI development teams</li>
    </ul>
    
    <h5>2. Transparency and Explainability</h5>
    <ul>
        <li>Making AI decision-making processes understandable</li>
        <li>Providing clear explanations for AI recommendations</li>
        <li>Documenting AI system capabilities and limitations</li>
        <li>Enabling human oversight and intervention</li>
    </ul>
    
    <h5>3. Privacy and Data Protection</h5>
    <ul>
        <li>Protecting individual privacy rights</li>
        <li>Implementing robust data security measures</li>
        <li>Obtaining informed consent for data use</li>
        <li>Minimizing data collection to what's necessary</li>
    </ul>
    
    <h5>4. Accountability and Responsibility</h5>
    <ul>
        <li>Clear ownership of AI system outcomes</li>
        <li>Mechanisms for addressing AI-related harms</li>
        <li>Regular monitoring and evaluation</li>
        <li>Continuous improvement and learning</li>
    </ul>
    
    <h3>AI Governance Frameworks: Building the Right Structure</h3>
    
    <p>Effective AI governance requires a comprehensive framework that addresses technical, organizational, and regulatory aspects of AI implementation.</p>
    
    <h4>Organizational AI Governance Structure</h4>
    
    <h5>AI Ethics Committee</h5>
    <ul>
        <li><strong>Composition:</strong> Cross-functional team including legal, technical, business, and ethics experts</li>
        <li><strong>Responsibilities:</strong> Review AI projects, establish ethical guidelines, monitor compliance</li>
        <li><strong>Authority:</strong> Approve or reject AI initiatives based on ethical criteria</li>
    </ul>
    
    <h5>AI Risk Management</h5>
    <ul>
        <li><strong>Risk Assessment:</strong> Regular evaluation of AI system risks and impacts</li>
        <li><strong>Mitigation Strategies:</strong> Proactive measures to address identified risks</li>
        <li><strong>Monitoring:</strong> Continuous surveillance of AI system behavior</li>
        <li><strong>Response Plans:</strong> Procedures for addressing AI-related incidents</li>
    </ul>
    
    <h3>Industry-Specific Ethical Considerations</h3>
    
    <p>Different industries face unique ethical challenges when implementing AI systems.</p>
    
    <h4>Healthcare AI Ethics</h4>
    <ul>
        <li><strong>Patient Safety:</strong> Ensuring AI recommendations don't harm patients</li>
        <li><strong>Medical Accuracy:</strong> Maintaining high standards of diagnostic and treatment quality</li>
        <li><strong>Informed Consent:</strong> Obtaining patient understanding of AI-assisted care</li>
        <li><strong>Professional Responsibility:</strong> Maintaining physician oversight and accountability</li>
    </ul>
    
    <h4>Financial Services AI Ethics</h4>
    <ul>
        <li><strong>Fair Lending:</strong> Preventing discriminatory lending practices</li>
        <li><strong>Transparency:</strong> Explaining credit decisions and risk assessments</li>
        <li><strong>Data Security:</strong> Protecting sensitive financial information</li>
        <li><strong>Market Integrity:</strong> Preventing AI-driven market manipulation</li>
    </ul>
    
    <h4>Manufacturing AI Ethics</h4>
    <ul>
        <li><strong>Worker Safety:</strong> Ensuring AI systems don't compromise workplace safety</li>
        <li><strong>Job Displacement:</strong> Managing the impact of automation on employment</li>
        <li><strong>Environmental Impact:</strong> Considering sustainability in AI implementation</li>
        <li><strong>Supply Chain Responsibility:</strong> Ensuring ethical practices throughout the supply chain</li>
    </ul>
    
    <h3>Implementation Framework for AI Ethics</h3>
    
    <h4>Phase 1: Foundation Building</h4>
    <ol>
        <li><strong>Establish AI Ethics Committee</strong>
            <ul>
                <li>Recruit diverse, qualified members</li>
                <li>Define roles and responsibilities</li>
                <li>Create decision-making processes</li>
            </ul>
        </li>
        <li><strong>Develop Ethical Guidelines</strong>
            <ul>
                <li>Define organizational AI values</li>
                <li>Create specific ethical standards</li>
                <li>Establish review procedures</li>
            </ul>
        </li>
        <li><strong>Implement Governance Structure</strong>
            <ul>
                <li>Create oversight mechanisms</li>
                <li>Establish reporting procedures</li>
                <li>Define accountability measures</li>
            </ul>
        </li>
    </ol>
    
    <h4>Phase 2: Process Integration</h4>
    <ol>
        <li><strong>Integrate Ethics into Development</strong>
            <ul>
                <li>Include ethics reviews in project planning</li>
                <li>Implement ethical testing procedures</li>
                <li>Create documentation requirements</li>
            </ul>
        </li>
        <li><strong>Train Development Teams</strong>
            <ul>
                <li>Provide ethics education and training</li>
                <li>Create awareness of bias and fairness issues</li>
                <li>Establish ethical decision-making skills</li>
            </ul>
        </li>
        <li><strong>Implement Monitoring Systems</strong>
            <ul>
                <li>Create continuous monitoring capabilities</li>
                <li>Establish alert systems for ethical violations</li>
                <li>Develop response procedures</li>
            </ul>
        </li>
    </ol>
    
    <h3>Best Practices for AI Ethics Implementation</h3>
    
    <h4>Technical Best Practices</h4>
    <ul>
        <li><strong>Diverse Training Data:</strong> Ensure representative and inclusive datasets</li>
        <li><strong>Bias Testing:</strong> Implement comprehensive bias detection and mitigation</li>
        <li><strong>Explainable AI:</strong> Use interpretable models and provide clear explanations</li>
        <li><strong>Human Oversight:</strong> Maintain human control over critical AI decisions</li>
    </ul>
    
    <h4>Organizational Best Practices</h4>
    <ul>
        <li><strong>Leadership Commitment:</strong> Ensure senior leadership supports ethical AI</li>
        <li><strong>Cross-functional Teams:</strong> Include diverse perspectives in AI development</li>
        <li><strong>Regular Training:</strong> Provide ongoing ethics education for all stakeholders</li>
        <li><strong>Transparent Communication:</strong> Share AI practices and outcomes openly</li>
    </ul>
    
    <h3>Measuring AI Ethics Success</h3>
    
    <h4>Key Performance Indicators</h4>
    <ul>
        <li><strong>Bias Reduction:</strong> Decrease in discriminatory outcomes over time</li>
        <li><strong>Transparency Score:</strong> Percentage of AI decisions that are explainable</li>
        <li><strong>Stakeholder Satisfaction:</strong> Feedback from affected communities</li>
        <li><strong>Compliance Rate:</strong> Adherence to ethical guidelines and regulations</li>
    </ul>
    
    <h3>Conclusion</h3>
    
    <p>AI ethics and governance are not optional—they are essential components of responsible AI implementation. Organizations that invest in ethical AI practices will not only avoid reputational and legal risks but will also build more trustworthy, effective, and sustainable AI systems.</p>
    
    <p>The key to success lies in treating AI ethics as a core business function, not an afterthought. By integrating ethical considerations into every aspect of AI development and deployment, organizations can harness the power of AI while maintaining trust, fairness, and human values.</p>
    
    <p>The future belongs to organizations that can successfully balance AI innovation with ethical responsibility. Those that get this balance right will not only succeed commercially but will also contribute to a more just and equitable world.</p>
</div>

<div class="page-break"></div>
'''
    
    # Find the Bibliography section and add Chapter 16 before it
    html_content = html_content.replace(
        '<div class="content">\nBibliography</h2>',
        chapter16_content + '\n<div class="content">\n<h2>Bibliography</h2>'
    )
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_WITH_CHAPTER16.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML with Chapter 16 created: {output_file}")
    return output_file

def generate_pdf_with_chapter16(html_file):
    """Generate PDF with Chapter 16"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_WITH_CHAPTER16.pdf'
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
            print(f"🎉 PDF with Chapter 16 generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Properly adding Chapter 16 to HTML...")
    
    # Add Chapter 16 properly
    html_file = properly_add_chapter16()
    
    if html_file:
        # Generate PDF with Chapter 16
        print("\n📚 Generating PDF with Chapter 16...")
        generate_pdf_with_chapter16(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 This PDF now includes:")
        print("1. ✅ All the working structure and formatting")
        print("2. ✅ Chapter 16: AI Ethics & Governance (properly added)")
        print("3. ✅ Complete ethical framework and implementation guide")
        print("4. ✅ All other enhancements")
        print("\n📖 The PDF should now have Chapter 16!")
    else:
        print("❌ Failed to add Chapter 16")
