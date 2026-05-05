// Render docs/results.html → docs/results.pdf using Playwright.
import { chromium } from 'playwright';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const htmlUrl = 'file://' + path.join(repoRoot, 'docs', 'results.html');
const outPdf = path.join(repoRoot, 'docs', 'results.pdf');

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1200, height: 1500 } });
const page = await ctx.newPage();
await page.goto(htmlUrl, { waitUntil: 'networkidle' });
// Expand all <details> so the full ResultJson dumps print too.
await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));
await page.emulateMedia({ media: 'print' });
await page.pdf({
  path: outPdf,
  format: 'A4',
  margin: { top: '15mm', right: '12mm', bottom: '15mm', left: '12mm' },
  printBackground: true,
});
await browser.close();
console.log('Wrote', outPdf);
