import os
import re
import subprocess
import markdown
import shutil

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(EDGE_PATH):
    EDGE_PATH = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
@page {
    size: A4;
    margin: 15mm 12mm 15mm 12mm;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    font-size: 13px;
    line-height: 1.55;
    color: #24292e;
    background-color: #fff;
    padding: 0;
    margin: 0;
}
h1, h2, h3, h4, h5, h6 {
    margin-top: 20px;
    margin-bottom: 12px;
    font-weight: 600;
    line-height: 1.25;
    color: #1a1f24;
    page-break-after: avoid;
}
h1 {
    font-size: 22px;
    padding-bottom: 8px;
    border-bottom: 2px solid #eaecef;
    margin-top: 0;
}
h2 {
    font-size: 16px;
    padding-bottom: 5px;
    border-bottom: 1px solid #eaecef;
}
h3 {
    font-size: 14px;
}
p, blockquote, ul, ol, dl, table, pre {
    margin-top: 0;
    margin-bottom: 12px;
}
code {
    font-family: "Cascadia Code", "Fira Code", Consolas, "Courier New", monospace;
    font-size: 11.5px;
    background-color: #f6f8fa;
    padding: 2px 5px;
    border-radius: 4px;
}
pre {
    background-color: #f6f8fa;
    border-radius: 6px;
    padding: 10px 12px;
    overflow: auto;
    font-size: 11.5px;
    line-height: 1.4;
    border: 1px solid #e1e4e8;
    page-break-inside: avoid;
}
pre code {
    background: none;
    padding: 0;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 16px;
    font-size: 11px;
    table-layout: auto;
    word-break: break-word;
    page-break-inside: avoid;
}
table th, table td {
    padding: 5px 6px;
    border: 1px solid #dfe2e5;
    text-align: center;
}
table th {
    font-weight: 600;
    background-color: #f6f8fa;
}
table tr:nth-child(2n) {
    background-color: #fafbfc;
}
blockquote {
    padding: 6px 12px;
    color: #57606a;
    border-left: 4px solid #0969da;
    background-color: #f6f8fa;
    margin: 10px 0;
    page-break-inside: avoid;
}
hr {
    height: 1px;
    background-color: #e1e4e8;
    border: none;
    margin: 16px 0;
}
ul, ol {
    padding-left: 22px;
}
li {
    margin-bottom: 3px;
}
.mermaid {
    text-align: center;
    margin: 15px 0;
    page-break-inside: avoid;
}
</style>
</head>
<body>
{content}
</body>
</html>
"""

def md_to_pdf(md_path, pdf_path):
    print(f"Converting {md_path} -> {pdf_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert markdown to html with tables and code extensions
    html_content = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"]
    )

    # Convert mermaid code blocks into <div class="mermaid">
    html_content = re.sub(
        r'<pre><code class="language-mermaid">([\s\S]*?)</code></pre>',
        r'<div class="mermaid">\1</div>',
        html_content
    )

    title = os.path.splitext(os.path.basename(md_path))[0]
    full_html = HTML_TEMPLATE.replace("{title}", title).replace("{content}", html_content)

    temp_html = md_path + ".temp.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    abs_html = os.path.abspath(temp_html).replace("\\", "/")
    cmd = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=3000",
        f"--print-to-pdf={os.path.abspath(pdf_path)}",
        f"file:///{abs_html}"
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(pdf_path):
        if os.path.exists(temp_html):
            os.remove(temp_html)
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"Success: {pdf_path} ({size_kb:.1f} KB)")
    else:
        print(f"Error converting {md_path}: {res.stderr.decode('utf-8', errors='ignore')}")

if __name__ == "__main__":
    import sys
    gi_dir = r"C:\Users\rsty\Desktop\GI"
    sub_dir = r"C:\Users\rsty\Desktop\GI\hard-label-dnn-extraction"

    files = [
        "CPU_EXPERIMENT_RESULTS.md",
        "SERVER_EXPERIMENT_RESULTS.md",
        "EXPERIMENT_OUTPUTS_GUIDE.md",
        "MODEL_SCALE_AND_RUNTIME_ANALYSIS.md",
    ]
    if len(sys.argv) >= 3:
        src = sys.argv[1]
        dst = sys.argv[2]
        md_to_pdf(src, dst)
    else:
        for f in files:
            m_path = os.path.join(gi_dir, f)
            p_name = os.path.splitext(f)[0] + ".pdf"
            p_path = os.path.join(gi_dir, p_name)
            if os.path.exists(m_path):
                md_to_pdf(m_path, p_path)
                # Sync both md and pdf to sub_dir
                shutil.copy2(m_path, os.path.join(sub_dir, f))
                if os.path.exists(p_path):
                    shutil.copy2(p_path, os.path.join(sub_dir, p_name))
