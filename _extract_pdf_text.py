from pathlib import Path
from pypdf import PdfReader

base = Path(r"C:\\Regular\\2026MCM")
files = [
    base / "2026_MCM_Problem_A.pdf",
    base / "2026_MCM_Problem_B.pdf",
    base / "2026_MCM_Problem_C.pdf",
    base / "Contest_AI_Policy.pdf",
]

out_dir = base / "_pdf_text"
out_dir.mkdir(exist_ok=True)

for pdf in files:
    reader = PdfReader(str(pdf))
    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.append(f"\n\n===== Page {i} =====\n\n{text}")
    out = out_dir / (pdf.stem + ".txt")
    out.write_text("\n".join(chunks), encoding="utf-8")
    print("wrote", out)
