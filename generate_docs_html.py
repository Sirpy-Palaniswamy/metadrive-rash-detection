"""
Converts PRESENTATION_professor.md and PROGRESS_REPORT.md into
print-ready HTML files.  Open each .html in Chrome/Edge and use
File > Print > Save as PDF  to get the final PDF.

Run:  python generate_docs_html.py
"""

import os
import markdown

BASE = os.path.dirname(os.path.abspath(__file__))

STYLE = """
<style>
  @page { margin: 20mm 18mm; }
  body {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12pt;
    line-height: 1.6;
    color: #1a1a2e;
    max-width: 900px;
    margin: 0 auto;
    padding: 12px 24px;
  }
  h1 {
    font-size: 22pt;
    color: #0f3460;
    border-bottom: 3px solid #0f3460;
    padding-bottom: 8px;
    margin-top: 32px;
  }
  h2 {
    font-size: 16pt;
    color: #16213e;
    border-bottom: 1px solid #aaa;
    padding-bottom: 4px;
    margin-top: 28px;
    page-break-after: avoid;
  }
  h3 {
    font-size: 13pt;
    color: #0f3460;
    margin-top: 20px;
    page-break-after: avoid;
  }
  h4 {
    font-size: 12pt;
    color: #333;
    margin-top: 16px;
  }
  /* Keep headings with following content */
  h1, h2, h3, h4 { page-break-inside: avoid; }
  p { margin: 6px 0 10px 0; }
  code {
    background: #f0f4f8;
    border: 1px solid #d0d8e4;
    border-radius: 3px;
    padding: 1px 5px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 10.5pt;
    color: #c0392b;
  }
  pre {
    background: #f6f8fa;
    border: 1px solid #d0d8e4;
    border-left: 4px solid #0f3460;
    border-radius: 4px;
    padding: 12px 16px;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 12px 0;
  }
  pre code {
    background: none;
    border: none;
    padding: 0;
    color: #1a1a2e;
    font-size: 9.5pt;
    line-height: 1.5;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0 16px 0;
    font-size: 11pt;
    page-break-inside: avoid;
  }
  th {
    background: #0f3460;
    color: #ffffff;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
  }
  td {
    padding: 6px 12px;
    border: 1px solid #c8d6e5;
  }
  tr:nth-child(even) td { background: #f0f4f8; }
  tr:hover td { background: #dce8f5; }
  blockquote {
    border-left: 4px solid #0f3460;
    background: #eef2f7;
    margin: 12px 0;
    padding: 10px 16px;
    font-style: italic;
    color: #333;
    border-radius: 0 4px 4px 0;
  }
  ul, ol { margin: 6px 0; padding-left: 24px; }
  li { margin: 3px 0; }
  hr {
    border: none;
    border-top: 2px solid #0f3460;
    margin: 28px 0;
  }
  /* Slide separator: triple-dash */
  .slide-break {
    border-top: 3px double #0f3460;
    margin: 36px 0 24px 0;
    page-break-before: always;
  }
  /* Cover banner */
  .cover {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    color: white;
    padding: 40px 36px;
    border-radius: 8px;
    margin-bottom: 32px;
  }
  .cover h1 {
    color: white;
    border-bottom: 2px solid rgba(255,255,255,0.4);
    font-size: 24pt;
  }
  .cover p { color: #c8d6e5; margin: 4px 0; }
  strong { color: #0f3460; }
  em { color: #444; }
</style>
"""

EXTENSIONS = [
    "tables",
    "fenced_code",
    "codehilite",
    "toc",
    "nl2br",
    "attr_list",
    "md_in_html",
]

def md_to_html(md_path: str, out_path: str, title: str):
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    md = markdown.Markdown(extensions=EXTENSIONS,
                           extension_configs={"codehilite": {"noclasses": True,
                                                              "pygments_style": "friendly"}})
    body = md.convert(text)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {STYLE}
</head>
<body>
{body}
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Written: {os.path.basename(out_path)}")


def main():
    docs = [
        (
            os.path.join(BASE, "PRESENTATION_professor.md"),
            os.path.join(BASE, "PRESENTATION_professor.html"),
            "Rash Driving Prediction — Professor Presentation",
        ),
        (
            os.path.join(BASE, "PROGRESS_REPORT.md"),
            os.path.join(BASE, "PROGRESS_REPORT.html"),
            "Rash Driving Prediction — Full Technical Progress Report",
        ),
    ]

    print("Generating HTML documents...")
    for md_path, html_path, title in docs:
        if not os.path.exists(md_path):
            print(f"  SKIP (not found): {md_path}")
            continue
        md_to_html(md_path, html_path, title)

    print()
    print("Done.  Next steps:")
    print("  1. Open PRESENTATION_professor.html  in Chrome / Edge")
    print("     File -> Print -> Save as PDF  (set margins to Minimum or None)")
    print("  2. Open PROGRESS_REPORT.html in Chrome / Edge")
    print("     File -> Print -> Save as PDF")
    print()
    print("Tip: In Chrome Print dialog, enable 'Background graphics' to keep")
    print("     the blue header bar on the cover sections.")


if __name__ == "__main__":
    main()
