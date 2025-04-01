# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "pdfplumber",
#     "pillow",
# ]
# ///


from pathlib import Path
from argparse import ArgumentParser
from datasets import load_dataset
import io
import PIL
from logging import getLogger
from typing import Optional
from datasets import Features, Sequence, Image, Value
from huggingface_hub import metadata_update
logger = getLogger(__name__)

features = Features(
    {
        "images": [Image()],
        "text": [Value("string")],
    }
)


def render(pdf):
    images = []
    for page in pdf.pages:
        buffer = io.BytesIO()
        page.to_image(resolution=200).save(buffer)
        image = PIL.Image.open(buffer)
        images.append(image)

    return images


def extract_text(pdf):
    logger.info("Extracting text from PDF pages")
    text = []
    text.extend(page.extract_text() for page in pdf.pages)
    return text


def list_pdfs_in_directory(directory: Path) -> list[Path]:
    logger.info(f"Scanning directory {directory} for PDF files")
    pdfs = list(directory.glob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDF files")
    return pdfs


def load_pdf_dataset(directory: Path):
    logger.info(f"Loading PDF dataset from directory {directory}")
    return load_dataset("pdffolder", data_dir=directory)


def prepare_dataset(
    directory: Path,
    hub_id: Optional[str] = None,
    private_repo: Optional[bool] = False,
    include_text: bool = True,
):
    logger.info(f"Preparing dataset from {directory} (include_text={include_text})")
    dataset = load_pdf_dataset(directory)

    if include_text:
        logger.info("Processing PDFs to extract images and text")
        dataset = dataset.map(
            lambda x: {
                "images": render(x["pdf"]),
                "text": extract_text(x["pdf"]),
            },
            remove_columns=["pdf"],
        )
    else:
        logger.info("Processing PDFs to extract images only")
        dataset = dataset.map(
            lambda x: {
                "images": render(x["pdf"]),
            },
            remove_columns=["pdf"],
            writer_batch_size=10,
        )
    dataset = dataset.cast_column("images", features["images"])
    logger.info("Dataset preparation completed")
    if hub_id:
        logger.info(f"Pushing dataset to hub {hub_id}")
        dataset.push_to_hub(hub_id, private=private_repo)
        metadata_update(hub_id, {"tags": "pdf"})
    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--extract-text", type=bool, default=True)
    parser.add_argument("--hub-id", type=str, required=True)
    parser.add_argument("--private-repo", type=bool, default=False)
    args = parser.parse_args()

    dataset = prepare_dataset(
        args.directory,
        args.hub_id,
        args.private_repo,
        args.extract_text,
    )

    print(dataset)
    print(dataset["train"].features)
