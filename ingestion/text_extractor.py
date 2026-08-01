"""
convert_one_pdf.py

Runs Docling on a SINGLE PDF file (full document, all pages) and saves
the result as markdown. Run this once per filing, as a separate process
each time — NOT looped over multiple files in one script — to avoid
memory accumulating across files (Docling's memory isn't fully released
between documents within the same process).

Usage:
    python convert_one_pdf.py "/path/to/apple_2024.pdf"
"""

import sys
import os
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat


def convert_pdf_to_markdown(pdf_path, output_dir):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist yet

    output_path = output_dir / (pdf_path.stem + ".md")
    # pdf_path.stem gives the filename without extension (e.g. "apple_2024"),
    # so output becomes "apple_2024.md" in the output folder

    print(f"Converting: {pdf_path.name}")
    print("This may take a while on the full document — do not interrupt.")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # NOTE: no page_range this time — full document, every page
    result = converter.convert(str(pdf_path))

    markdown_text = result.document.export_to_markdown()

    output_path.write_text(markdown_text, encoding="utf-8")

    print(f"Done. Saved to: {output_path}")
    print(f"Output length: {len(markdown_text):,} characters")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_one_pdf.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = "/Users/rajatthakral/Desktop/RAG_Project/extracted_text"

    convert_pdf_to_markdown(pdf_path, output_dir)