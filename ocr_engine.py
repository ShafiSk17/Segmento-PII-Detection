# ocr_engine.py
import pytesseract
from PIL import Image
import io

class OcrEngine:
    def __init__(self):
        """
        Initializes the OCR Engine using Tesseract.
        """
        self.available = False
        try:
            # Check availability by querying version
            pytesseract.get_tesseract_version()
            print("✅ Tesseract OCR Engine loaded.")
            self.available = True
        except Exception as e:
            print(f"❌ Tesseract OCR not found: {e}")
            print("👉 Install Tesseract system-wide (e.g., 'apt-get install tesseract-ocr') and 'pip install pytesseract'.")

    def extract_text(self, image_bytes: bytes) -> str:
        """
        Converts image bytes to text.
        """
        if not self.available:
            return ""
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Perform OCR
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"⚠️ OCR Extraction Error: {e}")
            return ""