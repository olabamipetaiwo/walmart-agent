"""
OCR Processor Module
Extracts item names and prices from receipt images using Tesseract/EasyOCR.
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# Conditional imports for OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


@dataclass
class ReceiptItem:
    """Represents an item extracted from a receipt."""
    name: str
    price: float
    quantity: int = 1
    raw_text: str = ""


@dataclass
class ReceiptData:
    """Parsed receipt data."""
    store_name: Optional[str]
    date: Optional[str]
    items: List[ReceiptItem]
    subtotal: Optional[float]
    tax: Optional[float]
    total: Optional[float]
    raw_text: str


class ReceiptOCRProcessor:
    """
    OCR processor for extracting structured data from receipt images.
    Supports both Tesseract and EasyOCR with automatic fallback.
    """
    
    def __init__(self, engine: str = "auto", language: str = "en"):
        """
        Initialize the OCR processor.
        
        Args:
            engine: OCR engine to use - "tesseract", "easyocr", or "auto"
            language: Language code for OCR
        """
        self.language = language
        self.engine = self._select_engine(engine)
        self.reader = None
        
        if self.engine == "easyocr" and EASYOCR_AVAILABLE:
            print("Initializing EasyOCR (this may take a moment)...")
            self.reader = easyocr.Reader([language], gpu=False)
            print("✓ EasyOCR initialized")
        elif self.engine == "tesseract" and TESSERACT_AVAILABLE:
            print("✓ Using Tesseract OCR")
        else:
            print("⚠ Running in mock mode (no OCR engine available)")
    
    def _select_engine(self, preference: str) -> str:
        """Select the best available OCR engine."""
        if preference == "auto":
            if EASYOCR_AVAILABLE:
                return "easyocr"
            elif TESSERACT_AVAILABLE:
                return "tesseract"
            else:
                return "mock"
        elif preference == "easyocr" and EASYOCR_AVAILABLE:
            return "easyocr"
        elif preference == "tesseract" and TESSERACT_AVAILABLE:
            return "tesseract"
        else:
            return "mock"
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess receipt image for better OCR accuracy.
        
        Args:
            image_path: Path to the receipt image.
            
        Returns:
            Preprocessed image as numpy array.
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding for better text contrast
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # Deskew if needed
        thresh = self._deskew(thresh)
        
        return thresh
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct skew in the receipt image.
        
        Args:
            image: Grayscale image.
            
        Returns:
            Deskewed image.
        """
        # Find all white pixels
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 10:
            return image
        
        # Get the minimum area rectangle
        try:
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Only correct if skew is significant
            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
        except:
            pass
        
        return image
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> str:
        """
        Extract raw text from a receipt image.
        
        Args:
            image_path: Path to the receipt image.
            preprocess: Whether to preprocess the image.
            
        Returns:
            Extracted text.
        """
        if self.engine == "mock":
            return self._mock_extract_text(image_path)
        
        if preprocess:
            img = self.preprocess_image(image_path)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if self.engine == "easyocr":
            return self._easyocr_extract(img)
        else:
            return self._tesseract_extract(img)
    
    def _tesseract_extract(self, image: np.ndarray) -> str:
        """Extract text using Tesseract."""
        # Tesseract config for receipt-style text
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(image, lang=self.language, config=config)
        return text
    
    def _easyocr_extract(self, image: np.ndarray) -> str:
        """Extract text using EasyOCR."""
        results = self.reader.readtext(image)
        # Combine all detected text
        lines = [result[1] for result in results]
        return "\n".join(lines)
    
    def _mock_extract_text(self, image_path: str) -> str:
        """Return mock receipt text for testing."""
        return """WALMART
Store #4523
123 Main Street
Anytown, USA 12345

Date: 01/20/2026   Time: 14:35

BANANAS           1.49
MILK 2% GAL       3.89
BREAD WHEAT       2.99
EGGS LARGE 12CT   4.29
CHICKEN BREAST    8.99
FROZEN PIZZA      7.49
DIAPERS HUGGIES  24.99
APPLE AIRPODS   149.99
HDMI CABLE        12.99
TOOTHPASTE        3.99

SUBTOTAL        221.10
TAX               17.69
TOTAL           238.79

CREDIT CARD      238.79

Thank you for shopping at Walmart!
"""
    
    def parse_receipt(self, image_path: str) -> ReceiptData:
        """
        Parse a receipt image and extract structured data.
        
        Args:
            image_path: Path to the receipt image.
            
        Returns:
            ReceiptData with parsed information.
        """
        raw_text = self.extract_text(image_path)
        return self.parse_text(raw_text)
    
    def parse_text(self, text: str) -> ReceiptData:
        """
        Parse raw receipt text into structured data.
        
        Args:
            text: Raw OCR text from receipt.
            
        Returns:
            ReceiptData with parsed information.
        """
        lines = text.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        
        items = []
        store_name = None
        date = None
        subtotal = None
        tax = None
        total = None
        
        # Price pattern: matches $X.XX or X.XX at end of line
        price_pattern = re.compile(r'(\d+\.\d{2})\s*$')
        
        # Item pattern: text followed by price
        item_pattern = re.compile(r'^(.+?)\s+(\d+\.\d{2})\s*$')
        
        # Date pattern
        date_pattern = re.compile(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})')
        
        for i, line in enumerate(lines):
            line_upper = line.upper()
            
            # Detect store name (usually first line)
            if i == 0 and not re.search(r'\d', line):
                store_name = line
                continue
            
            # Detect date
            date_match = date_pattern.search(line)
            if date_match and not date:
                date = date_match.group(1)
                continue
            
            # Skip non-item lines
            skip_keywords = ['STORE', 'STREET', 'ADDRESS', 'TIME', 'THANK', 
                           'CREDIT', 'DEBIT', 'CASH', 'CHANGE', 'CARD']
            if any(kw in line_upper for kw in skip_keywords):
                continue
            
            # Detect subtotal, tax, total
            if 'SUBTOTAL' in line_upper:
                match = price_pattern.search(line)
                if match:
                    subtotal = float(match.group(1))
                continue
            
            if 'TAX' in line_upper and 'SUBTOTAL' not in line_upper:
                match = price_pattern.search(line)
                if match:
                    tax = float(match.group(1))
                continue
            
            if 'TOTAL' in line_upper and 'SUBTOTAL' not in line_upper:
                match = price_pattern.search(line)
                if match:
                    total = float(match.group(1))
                continue
            
            # Try to parse as item line
            item_match = item_pattern.match(line)
            if item_match:
                name = item_match.group(1).strip()
                price = float(item_match.group(2))
                
                # Skip if it looks like a summary line
                if any(kw in name.upper() for kw in ['SUBTOTAL', 'TAX', 'TOTAL']):
                    continue
                
                # Detect quantity prefix (e.g., "2 x BANANAS")
                qty_match = re.match(r'^(\d+)\s*[xX]\s*(.+)$', name)
                if qty_match:
                    quantity = int(qty_match.group(1))
                    name = qty_match.group(2)
                else:
                    quantity = 1
                
                items.append(ReceiptItem(
                    name=name,
                    price=price,
                    quantity=quantity,
                    raw_text=line
                ))
        
        return ReceiptData(
            store_name=store_name,
            date=date,
            items=items,
            subtotal=subtotal,
            tax=tax,
            total=total,
            raw_text=text
        )
    
    def categorize_items(self, receipt: ReceiptData) -> Dict[str, List[ReceiptItem]]:
        """
        Categorize receipt items into spending categories.
        
        Args:
            receipt: Parsed receipt data.
            
        Returns:
            Dictionary mapping categories to items.
        """
        # Category keywords
        category_keywords = {
            "Groceries": ["MILK", "BREAD", "EGGS", "CHICKEN", "BEEF", "PORK", 
                         "BANANA", "APPLE", "ORANGE", "VEGETABLE", "FRUIT",
                         "CHEESE", "BUTTER", "YOGURT", "CEREAL", "RICE", "PASTA"],
            "Baby & Kids": ["DIAPER", "BABY", "FORMULA", "WIPES", "HUGGIES", 
                           "PAMPERS", "GERBER", "ENFAMIL"],
            "Electronics": ["AIRPOD", "CABLE", "HDMI", "USB", "CHARGER", 
                           "BATTERY", "HEADPHONE", "SPEAKER", "TV", "LAPTOP"],
            "Health & Beauty": ["TOOTHPASTE", "SHAMPOO", "SOAP", "LOTION",
                               "VITAMIN", "MEDICINE", "BANDAGE", "TYLENOL"],
            "Household": ["PAPER", "TOWEL", "CLEANER", "DETERGENT", "TRASH BAG",
                         "DISH", "SPONGE", "BLEACH"],
            "Frozen Foods": ["FROZEN", "PIZZA", "ICE CREAM", "POPSICLE"]
        }
        
        categorized = {cat: [] for cat in category_keywords}
        categorized["Other"] = []
        
        for item in receipt.items:
            item_upper = item.name.upper()
            matched = False
            
            for category, keywords in category_keywords.items():
                if any(kw in item_upper for kw in keywords):
                    categorized[category].append(item)
                    matched = True
                    break
            
            if not matched:
                categorized["Other"].append(item)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def get_receipt_summary(self, receipt: ReceiptData) -> Dict:
        """
        Generate a summary of the receipt.
        
        Args:
            receipt: Parsed receipt data.
            
        Returns:
            Dictionary with receipt summary.
        """
        categorized = self.categorize_items(receipt)
        
        category_totals = {}
        for cat, items in categorized.items():
            category_totals[cat] = sum(item.price * item.quantity for item in items)
        
        return {
            "store": receipt.store_name,
            "date": receipt.date,
            "item_count": len(receipt.items),
            "subtotal": receipt.subtotal,
            "tax": receipt.tax,
            "total": receipt.total,
            "categories": category_totals,
            "items": [
                {
                    "name": item.name,
                    "price": item.price,
                    "quantity": item.quantity
                }
                for item in receipt.items
            ]
        }


# Demo/test function
if __name__ == "__main__":
    processor = ReceiptOCRProcessor(engine="mock")
    
    print("Testing ReceiptOCRProcessor...")
    print("-" * 40)
    
    # Parse mock receipt
    receipt = processor.parse_receipt("sample_receipt.jpg")
    summary = processor.get_receipt_summary(receipt)
    
    print(f"Store: {summary['store']}")
    print(f"Date: {summary['date']}")
    print(f"Items: {summary['item_count']}")
    print(f"Total: ${summary['total']:.2f}")
    
    print("\nBy Category:")
    for cat, total in summary['categories'].items():
        print(f"  {cat}: ${total:.2f}")
    
    print("\nItems:")
    for item in summary['items']:
        print(f"  {item['name']}: ${item['price']:.2f}")
