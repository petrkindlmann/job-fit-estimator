// Render docs/explainer.html → docs/explainer.pdf using Playwright.
// Loads the file directly (no dev server) and waits for mermaid diagrams to render
// before printing.
import { chromium } from 'playwright';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const htmlUrl = 'file://' + path.join(repoRoot, 'docs', 'explainer.html');
const outPdf = path.join(repoRoot, 'docs', 'explainer.pdf');

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1024, height: 1400 } });
const page = await ctx.newPage();

await page.goto(htmlUrl, { waitUntil: 'networkidle' });

// Wait for mermaid diagrams to finish rendering (each .mermaid div should contain an <svg>).
await page.waitForFunction(() => {
  const els = document.querySelectorAll('.mermaid');
  if (!els.length) return true;
  return Array.from(els).every(el => el.querySelector('svg'));
}, { timeout: 15000 });

// Small extra delay to let mermaid finish layout passes.
await page.waitForTimeout(500);

await page.emulateMedia({ media: 'print' });
await page.pdf({
  path: outPdf,
  format: 'A4',
  margin: { top: '20mm', right: '15mm', bottom: '20mm', left: '15mm' },
  printBackground: true,
  preferCSSPageSize: false,
});

await browser.close();
console.log('Wrote', outPdf);
