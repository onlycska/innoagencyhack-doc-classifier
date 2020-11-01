from PIL import Image
import fitz
import logging


# noinspection PyUnresolvedReferences
def image_converting(path=None, image_body=None):
    images = []
    if path is not None:
        logging.debug(f"Image converting has started with 'path' parameter. Path: {path}")

        pdf_file = path
        doc = fitz.open(pdf_file)
        zoom = 200 / 72  # dpi parameter. standard is 72, but we need 200 for better text recognition.
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.getPixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

    elif image_body is not None:
        pass
    logging.debug(f"Images converted. There are {len(images)} pages.")
    return images
