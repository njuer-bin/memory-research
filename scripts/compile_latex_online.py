"""Compile paper/main.tex to PDF via LaTeX-on-HTTP (keyless public service).

Sends main.tex + refs.bib + acl.sty + acl_natbib.bst to
https://latex.ytotech.com/builds/sync and saves the resulting PDF to
paper/build/main.pdf. On non-PDF response, dumps the service log for
diagnosis.
"""
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"
OUT = PAPER / "build" / "main.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

FILES = ["main.tex", "refs.bib", "acl.sty", "acl_natbib.bst"]

def main() -> int:
    resources = [
        {"main": True, "content": (PAPER / "main.tex").read_text(encoding="utf-8")}
    ]
    for f in FILES[1:]:
        resources.append({"path": f, "content": (PAPER / f).read_text(encoding="utf-8")})
    payload = {
        "compiler": "pdflatex",
        "resources": resources,
        "options": {"compiler": {"bibliography": True}},
    }
    req = urllib.request.Request(
        "https://latex.ytotech.com/builds/sync",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            ct = resp.headers.get("Content-Type", "")
            data = resp.read()
            print("HTTP", resp.status, "| content-type:", ct)
            if "pdf" in ct.lower():
                OUT.write_bytes(data)
                print("PDF saved:", OUT, len(data), "bytes")
                return 0
            log = data.decode("utf-8", errors="replace")
            print("NOT PDF, body follows:")
            print(log[:4000])
            (PAPER / "build" / "latex-on-http-log.txt").write_text(log, encoding="utf-8")
            return 1
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print("HTTP error", e.code)
        print(body[:4000])
        (PAPER / "build" / "latex-on-http-log.txt").write_text(body, encoding="utf-8")
        return 1
    except Exception as e:
        print("request failed:", repr(e))
        return 2

if __name__ == "__main__":
    sys.exit(main())
