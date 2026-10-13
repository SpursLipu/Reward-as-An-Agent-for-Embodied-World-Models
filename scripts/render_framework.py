"""Render GitHub previews from the canonical paper framework PDF.

Install the optional renderer with: python -m pip install PyMuPDF
The PDF is the figure used in the paper; replace it when updating the figure.
"""
from pathlib import Path


def main():
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit("Install the renderer: python -m pip install PyMuPDF") from exc

    assets = Path(__file__).resolve().parents[1] / "assets"
    source = assets / "reward_as_agent_framework.pdf"
    with fitz.open(source) as document:
        if len(document) != 1:
            raise SystemExit("The framework PDF must contain exactly one page.")
        page = document[0]
        # Outline text so GitHub rendering does not depend on installed fonts.
        (assets / "reward_as_agent_framework.svg").write_text(
            page.get_svg_image(text_as_path=True), encoding="utf-8"
        )
        page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(
            assets / "reward_as_agent_framework.png"
        )
    print("Rendered SVG and PNG previews from the paper framework PDF.")


if __name__ == "__main__":
    main()
