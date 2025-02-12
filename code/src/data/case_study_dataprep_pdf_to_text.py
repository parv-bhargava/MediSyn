import os
import zipfile
from tempfile import TemporaryDirectory
from unstructured.partition.pdf import partition_pdf


def process_pdf_zip(zip_path: str, output_zip_path: str = "extracted_texts.zip"):
    """
    Process all PDFs in a ZIP file and save extracted texts into a new ZIP.

    Args:
        zip_path (str): Path to input ZIP file containing PDFs
        output_zip_path (str): Path for output ZIP file with text results
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_zip_path)
    if output_dir:  # Only create directories if path contains them
        os.makedirs(output_dir, exist_ok=True)

    with TemporaryDirectory() as tmp_dir:
        # Extract input ZIP to temporary directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Create output ZIP and process files
        with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        try:
                            # Extract text from PDF
                            elements = partition_pdf(
                                filename=pdf_path,
                                strategy="auto",
                                infer_table_structure=True
                            )
                            extracted_text = "\n\n".join([e.text for e in elements if e.text])

                            # Create relative path for text file in ZIP
                            relative_path = os.path.relpath(pdf_path, tmp_dir)
                            txt_path = os.path.splitext(relative_path)[0] + ".txt"

                            # Add to output ZIP
                            output_zip.writestr(txt_path, extracted_text)
                            print(f"Processed: {relative_path} → {txt_path}")

                        except Exception as e:
                            print(f"Error processing {file}: {str(e)}")

os.chdir("../../../../Case_Study_Data")
process_pdf_zip(
    zip_path="case-study.zip",  # Your input ZIP path
    output_zip_path="case-study-text.zip"  # Output in current directory
)