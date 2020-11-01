import asyncio
import cv2
import numpy as np
import pytesseract
import logging

from PIL import Image

from model.db_requests import Database
from postprocessing.json_formatter import create_json_ans
from preprocessing.image_converting import image_converting
from preprocessing.workers import extract_text_worker


class TextExtractor:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        self.img_body: cv2 = None

    @staticmethod
    def recognizing(img_body):
        text = pytesseract.image_to_string(img_body, lang="rus")
        return text

    def image_normalizing(self, delay=1):
        if self.img_body is not None:
            # --- dilation on the green channel ---
            dilated_img = cv2.dilate(self.img_body[:, :, 1], np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)

            # --- finding absolute difference to preserve edges ---
            diff_img = 255 - cv2.absdiff(self.img_body[:, :, 1], bg_img)

            # --- normalizing between 0 to 255 ---
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            return norm_img
        else:
            logging.error("self.img_body parameter is None")

    def text_extraction_pipeline(self, img_body: cv2):
        self.img_body = img_body
        # await self.image_preprocessing()
        open_cv_image = self.image_normalizing()
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        open_cv_image = Image.fromarray(open_cv_image)
        text = self.recognizing(open_cv_image)
        return text

    @staticmethod
    def doc_worker(filepath, doc_type=None):
        pages = {}
        if filepath.lower().endswith(".pdf"):
            images = image_converting(filepath)
            if images:
                images_count = len(images)
                for i in range(images_count):
                    page_text = extract_text_worker(images[i])
                    page_text = page_text.replace("..", ".").strip().split()
                    page_text = " ".join(page_text)
                    pages[i + 1] = page_text
                text_to_db = " ".join(pages.values())
            else:
                logging.error("Images wasn't converted. Variable 'images' is empty.")
                return None
        else:
            images_count = 1
            img = Image.open(filepath)
            page_text = extract_text_worker(img)
            pages[1] = page_text
            text_to_db = page_text
        # TODO проставляются лишние табы в результате, пока непонятно почему
        Database().extracted_data_insert(text_to_db, doc_type)
        logging.debug("Extracted text inserted in database")
        json = create_json_ans(params=pages, pages_count=images_count, full_text=text_to_db)
        return json
