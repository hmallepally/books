const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

async function buildBook() {
    console.log('Starting build process for Theories and Models of Chris Argyris and Peter Senge...');
    const chaptersDir = path.join(__dirname, 'chapters');
    const visualsDir = path.join(__dirname, 'visuals');
    const buildDir = path.join(__dirname, '_build');

    if (!fs.existsSync(buildDir)) fs.mkdirSync(buildDir);
    if (!fs.existsSync(visualsDir)) fs.mkdirSync(visualsDir);

    const files = fs.readdirSync(chaptersDir)
        .filter(f => f.endsWith('.md'))
        .sort();

    console.log(`Found ${files.length} chapters.`);

    let fullManuscript = '';
    let epubManuscript = '';
    let diagramCount = 0;

    // Visual map: chapter file → [visual filename, alt text]
    const visualMap = {
        '08-ch5-argyris-personal.md': ['chris_argyris.png', 'Portrait of Chris Argyris, HBS Professor and pioneer of double-loop learning'],
        '13-ch10-senge-personal.md': ['peter_senge.png', 'Portrait of Peter Senge, MIT Senior Lecturer and developer of Systems Thinking']
    };

    // Alt text for Mermaid diagrams (if any exist in chapters)
    const diagramAltTexts = [];

    for (const file of files) {
        let content = fs.readFileSync(path.join(chaptersDir, file), 'utf-8');

        // Strip \newpage commands — CSS handles page breaks
        content = content.replace(/\\newpage\r?\n?/g, '');

        // Process Mermaid blocks (shared — generates diagram PNGs)
        const mermaidRegex = /```mermaid\r?\n([\s\S]*?)```/g;
        content = content.replace(mermaidRegex, (match, code) => {
            diagramCount++;
            const mmdPath = path.join(buildDir, `diagram_${diagramCount}.mmd`);
            const pngPath = path.join(visualsDir, `diagram_${diagramCount}.png`);

            fs.writeFileSync(mmdPath, code);
            console.log(`Generating diagram ${diagramCount}...`);

            try {
                execSync(`mmdc -i "${mmdPath}" -o "${pngPath}" -b transparent -t dark -s 3 -w 1200`, { stdio: 'inherit' });
            } catch (err) {
                console.error(`Failed to generate diagram ${diagramCount}`, err);
            }
            return `![__DIAGRAM_${diagramCount}__](${pngPath})`;
        });

        // Convert footnotes: [^N] → <sup>N</sup>
        content = content.replace(/\[\^(\d+)\](?!:)/g, '<sup>$1</sup>');
        content = content.replace(/\[\^(\d+)\]:\s*(.*)/g, '<small><sup>$1</sup> $2</small>');

        // --- PDF version: empty alt text (no visible captions) ---
        let pdfContent = content;
        if (visualMap[file]) {
            const [visualFile] = visualMap[file];
            const visualPath = `visuals/${visualFile}`;
            if (fs.existsSync(path.join(__dirname, visualPath))) {
                pdfContent = pdfContent.replace(/^(# .*\r?\n+)/m, `$1![](${visualPath})\n\n`);
            }
        }
        pdfContent = pdfContent.replace(/!\[__DIAGRAM_\d+__\]/g, '![]');
        fullManuscript += `\n\n${pdfContent}`;

        // --- EPUB version: descriptive alt text (accessibility) ---
        if (file !== '02-toc.md') {
            let epubContent = content;
            if (visualMap[file]) {
                const [visualFile, altText] = visualMap[file];
                const visualPath = `visuals/${visualFile}`;
                if (fs.existsSync(path.join(__dirname, visualPath))) {
                    epubContent = epubContent.replace(/^(# .*\r?\n+)/m, `$1![${altText}](${visualPath})\n\n`);
                }
            }
            epubContent = epubContent.replace(/!\[__DIAGRAM_(\d+)__\]/g, (match, num) => {
                const idx = parseInt(num) - 1;
                const altText = diagramAltTexts[idx] || `Systems diagram figure ${num}`;
                return `![${altText}]`;
            });
            epubManuscript += `\n\n${epubContent}`;
        }
    }

    const manuscriptPath = path.join(buildDir, 'manuscript.md');
    fs.writeFileSync(manuscriptPath, fullManuscript);

    const epubManuscriptPath = path.join(buildDir, 'manuscript_epub.md');
    fs.writeFileSync(epubManuscriptPath, epubManuscript);
    console.log('Combined manuscripts written (PDF + EPUB).');

    // Generate EPUB
    console.log('Generating EPUB...');
    try {
        const epubName = "Theories_and_Models_of_Chris_Argyris_and_Peter_Senge.epub";
        const coverFlag = fs.existsSync(path.join(visualsDir, 'cover.png')) ? '--epub-cover-image=visuals/cover.png' : '';
        execSync(`pandoc metadata.yaml _build/manuscript_epub.md -o "${epubName}" --toc --toc-depth=1 --css=epub.css ${coverFlag}`, { stdio: 'inherit' });
        console.log('EPUB generated successfully.');
    } catch (err) {
        console.error('Failed to generate EPUB', err);
    }

    // Generate HTML for PDF
    console.log('Generating HTML for PDF...');
    const htmlPath = path.join(buildDir, 'manuscript.html');
    try {
        execSync(`pandoc metadata.yaml _build/manuscript.md -o _build/manuscript.html --standalone --embed-resources --css=style.css`, { stdio: 'inherit' });
        console.log('HTML generated successfully.');
    } catch (err) {
        console.error('Failed to generate HTML', err);
    }

    // Generate PDF via Headless Chrome
    console.log('Generating PDF...');
    try {
        const chromePath = `"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"`;
        const pdfPath = path.join(__dirname, 'Theories_and_Models_of_Chris_Argyris_and_Peter_Senge.pdf');
        const fileHtmlPath = htmlPath.replace(/\\/g, '/');
        execSync(`${chromePath} --headless=new --print-to-pdf="${pdfPath}" --no-pdf-header-footer "file:///${fileHtmlPath}"`, { stdio: 'inherit' });
        console.log('PDF generated successfully.');

        // Post-process: stamp page numbers onto the PDF
        console.log('Adding page numbers...');
        const stampScript = path.join(__dirname, 'stamp_pages.py');
        execSync(`python "${stampScript}" "${pdfPath}"`, { stdio: 'inherit' });
        console.log('Page numbers added successfully.');
    } catch (err) {
        console.error('Failed to generate PDF.', err.message);
    }
}

buildBook().catch(console.error);
