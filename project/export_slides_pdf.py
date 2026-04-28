"""Export marimo slide view to a multi-page PDF.

Navigates through each slide by pressing ArrowRight in the embedded
reveal.js deck, screenshots each one, and stitches into a PDF.

Usage:
    python export_slides_pdf.py [url] [output.pdf] [wait_seconds]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from PIL import Image


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/?mode=present"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "notebook_slides.pdf")
    kernel_wait = int(sys.argv[3]) if len(sys.argv) > 3 else 90

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=180_000)
        print(f"loaded {url}; waiting {kernel_wait}s for pyodide + cells")
        page.wait_for_selector(".reveal .slides section", timeout=180_000)
        time.sleep(kernel_wait)

        # Force slides container to fill viewport (reveal renders embedded)
        page.evaluate(
            """
            () => {
              const rev = document.querySelector('.reveal');
              if (!rev) return;
              const parent = rev.parentElement;
              Object.assign(rev.style, {position:'fixed',inset:'0',width:'100vw',height:'100vh',zIndex:'99999',background:'white',border:'none',margin:'0'});
              // hide app chrome
              document.querySelectorAll('header, nav, [role=toolbar], .mo-toolbar').forEach(el => el.style.display='none');
            }
            """
        )
        time.sleep(1)

        total = page.evaluate("document.querySelectorAll('.reveal .slides > section').length")
        print(f"slides detected: {total}")

        # Click the reveal container to give it focus (keyboard nav)
        page.evaluate("document.querySelector('.reveal').click();")
        time.sleep(0.3)

        images = []
        for i in range(total):
            time.sleep(1.0)
            png = f"/tmp/slide_{i:03d}.png"
            # Screenshot just the reveal viewport
            elt = page.query_selector(".reveal")
            if elt:
                elt.screenshot(path=png)
            else:
                page.screenshot(path=png, full_page=False)
            images.append(png)
            print(f"  {i+1}/{total}")
            # advance
            page.keyboard.press("ArrowRight")

        context.close()
        browser.close()

    ims = [Image.open(p).convert("RGB") for p in images]
    ims[0].save(out, save_all=True, append_images=ims[1:])
    print(f"wrote {out} with {len(ims)} pages")


if __name__ == "__main__":
    main()
