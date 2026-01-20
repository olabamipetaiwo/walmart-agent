"""
Vision Engine Module
Uses YOLOv8 for shopping cart item detection and classification.
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Conditional import for ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Using mock detection.")


@dataclass
class DetectedItem:
    """Represents a detected item in the shopping cart."""
    name: str
    category: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    estimated_price: Optional[float] = None


# Mapping from COCO classes to Walmart categories
COCO_TO_WALMART_MAPPING = {
    # Electronics
    "cell phone": ("Smartphone", "Electronics"),
    "laptop": ("Laptop", "Electronics"),
    "tv": ("Television", "Electronics"),
    "remote": ("Remote Control", "Electronics"),
    "keyboard": ("Keyboard", "Electronics"),
    "mouse": ("Computer Mouse", "Electronics"),
    
    # Food & Groceries
    "banana": ("Bananas", "Groceries"),
    "apple": ("Apples", "Groceries"),
    "orange": ("Oranges", "Groceries"),
    "sandwich": ("Deli Sandwich", "Groceries"),
    "pizza": ("Frozen Pizza", "Groceries"),
    "donut": ("Donuts", "Groceries"),
    "cake": ("Bakery Cake", "Groceries"),
    "carrot": ("Carrots", "Groceries"),
    "broccoli": ("Broccoli", "Groceries"),
    "hot dog": ("Hot Dogs", "Groceries"),
    
    # Household
    "bottle": ("Water Bottle", "Household"),
    "cup": ("Cups", "Household"),
    "bowl": ("Bowls", "Household"),
    "knife": ("Kitchen Knife", "Household"),
    "spoon": ("Spoons", "Household"),
    "fork": ("Forks", "Household"),
    "toothbrush": ("Toothbrush", "Health & Beauty"),
    "scissors": ("Scissors", "Household"),
    "clock": ("Wall Clock", "Household"),
    "vase": ("Decorative Vase", "Household"),
    
    # Sports & Outdoors
    "sports ball": ("Sports Ball", "Sports"),
    "tennis racket": ("Tennis Racket", "Sports"),
    "baseball bat": ("Baseball Bat", "Sports"),
    "baseball glove": ("Baseball Glove", "Sports"),
    "skateboard": ("Skateboard", "Sports"),
    "surfboard": ("Surfboard", "Sports"),
    "frisbee": ("Frisbee", "Sports"),
    "skis": ("Skis", "Sports"),
    "snowboard": ("Snowboard", "Sports"),
    
    # Baby & Kids
    "teddy bear": ("Stuffed Toy", "Baby & Kids"),
    
    # Clothing
    "backpack": ("Backpack", "Clothing"),
    "umbrella": ("Umbrella", "Clothing"),
    "handbag": ("Handbag", "Clothing"),
    "tie": ("Necktie", "Clothing"),
    "suitcase": ("Suitcase", "Clothing"),
    
    # Books & Media
    "book": ("Book", "Books & Media"),
}

# Estimated prices for demo purposes
ESTIMATED_PRICES = {
    "Smartphone": 299.99,
    "Laptop": 549.99,
    "Television": 399.99,
    "Remote Control": 24.99,
    "Keyboard": 49.99,
    "Computer Mouse": 29.99,
    "Bananas": 1.49,
    "Apples": 3.99,
    "Oranges": 4.49,
    "Deli Sandwich": 6.99,
    "Frozen Pizza": 7.49,
    "Donuts": 5.99,
    "Bakery Cake": 14.99,
    "Carrots": 2.49,
    "Broccoli": 2.99,
    "Hot Dogs": 4.99,
    "Water Bottle": 12.99,
    "Cups": 8.99,
    "Bowls": 9.99,
    "Kitchen Knife": 19.99,
    "Spoons": 6.99,
    "Forks": 6.99,
    "Toothbrush": 4.99,
    "Scissors": 7.99,
    "Wall Clock": 24.99,
    "Decorative Vase": 19.99,
    "Sports Ball": 19.99,
    "Tennis Racket": 79.99,
    "Baseball Bat": 34.99,
    "Baseball Glove": 49.99,
    "Skateboard": 59.99,
    "Surfboard": 299.99,
    "Frisbee": 9.99,
    "Skis": 399.99,
    "Snowboard": 299.99,
    "Stuffed Toy": 14.99,
    "Backpack": 39.99,
    "Umbrella": 19.99,
    "Handbag": 49.99,
    "Necktie": 24.99,
    "Suitcase": 89.99,
    "Book": 14.99,
}


class CartVisionEngine:
    """
    Computer Vision engine for detecting items in shopping cart images.
    Uses YOLOv8 pre-trained on COCO dataset with custom mapping to retail items.
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25):
        """
        Initialize the vision engine.
        
        Args:
            model_path: Path to custom YOLO weights. If None, uses YOLOv8n.
            confidence_threshold: Minimum confidence for detections (0-1).
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if YOLO_AVAILABLE:
            try:
                if model_path and os.path.exists(model_path):
                    self.model = YOLO(model_path)
                else:
                    # Use pre-trained YOLOv8 nano (fastest, good for demo)
                    self.model = YOLO("yolov8n.pt")
                self.model.to(self.device)
                print(f"✓ YOLO model loaded on {self.device}")
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")
                self.model = None
        else:
            print("⚠ Running in mock mode (ultralytics not installed)")
    
    def detect_items(self, image_path: str) -> List[DetectedItem]:
        """
        Detect items in a shopping cart image.
        
        Args:
            image_path: Path to the cart image.
            
        Returns:
            List of DetectedItem objects.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if self.model is None:
            return self._mock_detection(image_path)
        
        return self._yolo_detection(image_path)
    
    def _yolo_detection(self, image_path: str) -> List[DetectedItem]:
        """Run actual YOLO detection."""
        results = self.model(image_path, verbose=False)
        detected_items = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])
                if confidence < self.confidence_threshold:
                    continue
                
                class_id = int(boxes.cls[i])
                class_name = self.model.names[class_id].lower()
                
                # Map COCO class to Walmart item
                if class_name in COCO_TO_WALMART_MAPPING:
                    item_name, category = COCO_TO_WALMART_MAPPING[class_name]
                else:
                    item_name = class_name.title()
                    category = "General"
                
                # Get bounding box
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Estimate price
                price = ESTIMATED_PRICES.get(item_name, 9.99)
                
                detected_items.append(DetectedItem(
                    name=item_name,
                    category=category,
                    confidence=confidence,
                    bounding_box=(x1, y1, x2, y2),
                    estimated_price=price
                ))
        
        return detected_items
    
    def _mock_detection(self, image_path: str) -> List[DetectedItem]:
        """
        Mock detection for testing when YOLO is not available.
        Returns sample items based on image filename hints.
        """
        filename = os.path.basename(image_path).lower()
        
        # Default mock items
        mock_items = [
            DetectedItem(
                name="Laptop",
                category="Electronics",
                confidence=0.92,
                bounding_box=(50, 50, 300, 250),
                estimated_price=549.99
            ),
            DetectedItem(
                name="Bananas",
                category="Groceries",
                confidence=0.88,
                bounding_box=(320, 100, 450, 200),
                estimated_price=1.49
            ),
            DetectedItem(
                name="Toothbrush",
                category="Health & Beauty",
                confidence=0.85,
                bounding_box=(100, 280, 180, 350),
                estimated_price=4.99
            ),
        ]
        
        # Add variety based on filename
        if "electronics" in filename:
            mock_items.append(DetectedItem(
                name="Television",
                category="Electronics",
                confidence=0.91,
                bounding_box=(200, 50, 500, 350),
                estimated_price=399.99
            ))
        elif "baby" in filename:
            mock_items.append(DetectedItem(
                name="Stuffed Toy",
                category="Baby & Kids",
                confidence=0.89,
                bounding_box=(150, 150, 280, 300),
                estimated_price=14.99
            ))
        elif "groceries" in filename:
            mock_items.extend([
                DetectedItem(
                    name="Apples",
                    category="Groceries",
                    confidence=0.87,
                    bounding_box=(50, 100, 150, 180),
                    estimated_price=3.99
                ),
                DetectedItem(
                    name="Frozen Pizza",
                    category="Groceries",
                    confidence=0.84,
                    bounding_box=(200, 120, 350, 220),
                    estimated_price=7.49
                ),
            ])
        
        return mock_items
    
    def annotate_image(self, image_path: str, items: List[DetectedItem]) -> Image.Image:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image_path: Path to the original image.
            items: List of detected items.
            
        Returns:
            Annotated PIL Image.
        """
        from PIL import ImageDraw, ImageFont
        
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fallback to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()
        
        # Color mapping for categories
        category_colors = {
            "Electronics": "#FF6B6B",
            "Groceries": "#4ECDC4",
            "Household": "#45B7D1",
            "Health & Beauty": "#96CEB4",
            "Sports": "#FFEAA7",
            "Baby & Kids": "#DDA0DD",
            "Clothing": "#98D8C8",
            "Books & Media": "#F7DC6F",
            "General": "#BDC3C7",
        }
        
        for item in items:
            x1, y1, x2, y2 = item.bounding_box
            color = category_colors.get(item.category, "#BDC3C7")
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            label = f"{item.name} (${item.estimated_price:.2f})"
            bbox = draw.textbbox((x1, y1 - 20), label, font=font)
            draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
            
            # Draw label text
            draw.text((x1, y1 - 20), label, fill="white", font=font)
        
        return img
    
    def get_cart_summary(self, items: List[DetectedItem]) -> Dict:
        """
        Generate a summary of the detected cart items.
        
        Args:
            items: List of detected items.
            
        Returns:
            Dictionary with cart summary.
        """
        total = sum(item.estimated_price or 0 for item in items)
        
        # Group by category
        categories = {}
        for item in items:
            if item.category not in categories:
                categories[item.category] = {"items": [], "subtotal": 0}
            categories[item.category]["items"].append(item.name)
            categories[item.category]["subtotal"] += item.estimated_price or 0
        
        return {
            "total_items": len(items),
            "estimated_total": round(total, 2),
            "categories": categories,
            "items": [
                {
                    "name": item.name,
                    "category": item.category,
                    "price": item.estimated_price,
                    "confidence": round(item.confidence, 2)
                }
                for item in items
            ]
        }


# Demo/test function
if __name__ == "__main__":
    engine = CartVisionEngine()
    
    # Test with a mock image path
    print("Testing CartVisionEngine...")
    print("-" * 40)
    
    # Mock test
    items = engine._mock_detection("sample_groceries_cart.jpg")
    summary = engine.get_cart_summary(items)
    
    print(f"Detected {summary['total_items']} items")
    print(f"Estimated Total: ${summary['estimated_total']:.2f}")
    print("\nBy Category:")
    for cat, data in summary['categories'].items():
        print(f"  {cat}: ${data['subtotal']:.2f}")
        for item in data['items']:
            print(f"    - {item}")
