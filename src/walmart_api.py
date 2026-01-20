"""
Walmart Mock API Module
Simulates Walmart's product database for price lookups and category info.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from fuzzywuzzy import fuzz, process


@dataclass
class Product:
    """Represents a Walmart product."""
    id: str
    name: str
    category: str
    price: float
    bnpl_eligible: bool
    department: str
    brand: Optional[str] = None


# Mock product catalog
PRODUCT_CATALOG = [
    # Electronics
    Product("EL001", "Apple AirPods Pro", "Electronics", 249.99, True, "Electronics", "Apple"),
    Product("EL002", "Samsung Galaxy Buds", "Electronics", 149.99, True, "Electronics", "Samsung"),
    Product("EL003", "Sony PlayStation 5", "Electronics", 499.99, True, "Electronics", "Sony"),
    Product("EL004", "Nintendo Switch", "Electronics", 299.99, True, "Electronics", "Nintendo"),
    Product("EL005", "Apple iPad 10th Gen", "Electronics", 449.99, True, "Electronics", "Apple"),
    Product("EL006", "HDMI Cable 6ft", "Electronics", 12.99, True, "Electronics", "Insignia"),
    Product("EL007", "USB-C Charger", "Electronics", 24.99, True, "Electronics", "Anker"),
    Product("EL008", "Wireless Mouse", "Electronics", 29.99, True, "Electronics", "Logitech"),
    Product("EL009", "Mechanical Keyboard", "Electronics", 79.99, True, "Electronics", "Corsair"),
    Product("EL010", "65-inch Smart TV", "Electronics", 548.00, True, "Electronics", "TCL"),
    Product("EL011", "Laptop HP 15.6in", "Electronics", 549.99, True, "Electronics", "HP"),
    Product("EL012", "Bluetooth Speaker", "Electronics", 39.99, True, "Electronics", "JBL"),
    
    # Groceries (Not BNPL eligible typically)
    Product("GR001", "Bananas per lb", "Groceries", 0.58, False, "Produce", "Fresh"),
    Product("GR002", "Apples Gala 3lb", "Groceries", 4.97, False, "Produce", "Fresh"),
    Product("GR003", "Milk 2% Gallon", "Groceries", 3.89, False, "Dairy", "Great Value"),
    Product("GR004", "Eggs Large 12ct", "Groceries", 4.29, False, "Dairy", "Great Value"),
    Product("GR005", "Bread Wheat", "Groceries", 2.99, False, "Bakery", "Nature's Own"),
    Product("GR006", "Chicken Breast 2lb", "Groceries", 8.99, False, "Meat", "Tyson"),
    Product("GR007", "Ground Beef 1lb", "Groceries", 5.99, False, "Meat", "80/20"),
    Product("GR008", "Frozen Pizza", "Groceries", 7.49, False, "Frozen", "DiGiorno"),
    Product("GR009", "Orange Juice 52oz", "Groceries", 4.48, False, "Beverages", "Tropicana"),
    Product("GR010", "Cereal Cheerios", "Groceries", 4.99, False, "Breakfast", "General Mills"),
    Product("GR011", "Rice 5lb Bag", "Groceries", 5.49, False, "Pantry", "Mahatma"),
    Product("GR012", "Pasta Spaghetti", "Groceries", 1.49, False, "Pantry", "Barilla"),
    
    # Baby & Kids
    Product("BY001", "Huggies Diapers Size 3", "Baby & Kids", 24.99, True, "Baby", "Huggies"),
    Product("BY002", "Pampers Swaddlers", "Baby & Kids", 27.99, True, "Baby", "Pampers"),
    Product("BY003", "Baby Wipes 800ct", "Baby & Kids", 19.99, True, "Baby", "Huggies"),
    Product("BY004", "Enfamil Formula 20.4oz", "Baby & Kids", 44.99, True, "Baby", "Enfamil"),
    Product("BY005", "Baby Bottle Set", "Baby & Kids", 15.99, True, "Baby", "Philips Avent"),
    Product("BY006", "Stuffed Teddy Bear", "Baby & Kids", 14.99, True, "Toys", "Build-A-Bear"),
    Product("BY007", "LEGO Classic Set", "Baby & Kids", 34.99, True, "Toys", "LEGO"),
    Product("BY008", "Kids Tablet", "Baby & Kids", 89.99, True, "Toys", "Amazon Fire"),
    
    # Health & Beauty
    Product("HB001", "Toothpaste Crest", "Health & Beauty", 3.99, False, "Oral Care", "Crest"),
    Product("HB002", "Shampoo Head & Shoulders", "Health & Beauty", 7.99, False, "Hair Care", "Head & Shoulders"),
    Product("HB003", "Body Wash Dove", "Health & Beauty", 6.99, False, "Body Care", "Dove"),
    Product("HB004", "Tylenol Extra Strength", "Health & Beauty", 9.99, False, "Medicine", "Tylenol"),
    Product("HB005", "Vitamins Multivitamin", "Health & Beauty", 12.99, False, "Vitamins", "Nature Made"),
    Product("HB006", "Deodorant Old Spice", "Health & Beauty", 6.49, False, "Personal Care", "Old Spice"),
    Product("HB007", "Electric Toothbrush", "Health & Beauty", 49.99, True, "Oral Care", "Oral-B"),
    Product("HB008", "Hair Dryer", "Health & Beauty", 29.99, True, "Hair Care", "Revlon"),
    
    # Household
    Product("HH001", "Paper Towels 8pk", "Household", 15.99, False, "Paper Goods", "Bounty"),
    Product("HH002", "Toilet Paper 24pk", "Household", 22.99, False, "Paper Goods", "Charmin"),
    Product("HH003", "Laundry Detergent", "Household", 11.99, False, "Cleaning", "Tide"),
    Product("HH004", "Dish Soap", "Household", 3.99, False, "Cleaning", "Dawn"),
    Product("HH005", "Trash Bags 50ct", "Household", 12.99, False, "Cleaning", "Glad"),
    Product("HH006", "All-Purpose Cleaner", "Household", 4.99, False, "Cleaning", "Lysol"),
    Product("HH007", "Vacuum Cleaner", "Household", 199.99, True, "Appliances", "Shark"),
    Product("HH008", "Air Fryer", "Household", 89.99, True, "Appliances", "Ninja"),
    Product("HH009", "Coffee Maker", "Household", 79.99, True, "Appliances", "Keurig"),
    Product("HH010", "Microwave", "Household", 119.99, True, "Appliances", "Hamilton Beach"),
    
    # Clothing
    Product("CL001", "Men's T-Shirt", "Clothing", 8.99, True, "Men's", "Fruit of the Loom"),
    Product("CL002", "Women's Jeans", "Clothing", 24.99, True, "Women's", "Time and Tru"),
    Product("CL003", "Kids Sneakers", "Clothing", 19.99, True, "Kids", "Athletic Works"),
    Product("CL004", "Winter Jacket", "Clothing", 49.99, True, "Outerwear", "Swiss Tech"),
    Product("CL005", "Backpack", "Clothing", 29.99, True, "Accessories", "Ozark Trail"),
    
    # Sports & Outdoors
    Product("SP001", "Basketball", "Sports", 24.99, True, "Team Sports", "Spalding"),
    Product("SP002", "Yoga Mat", "Sports", 19.99, True, "Fitness", "CAP"),
    Product("SP003", "Dumbbells 20lb Set", "Sports", 49.99, True, "Fitness", "CAP"),
    Product("SP004", "Camping Tent 4-Person", "Sports", 89.99, True, "Camping", "Ozark Trail"),
    Product("SP005", "Fishing Rod Combo", "Sports", 34.99, True, "Fishing", "Zebco"),
    
    # Furniture
    Product("FN001", "Office Chair", "Furniture", 149.99, True, "Office", "Mainstays"),
    Product("FN002", "Bookshelf", "Furniture", 79.99, True, "Living Room", "Better Homes"),
    Product("FN003", "TV Stand", "Furniture", 129.99, True, "Living Room", "Mainstays"),
    Product("FN004", "Bed Frame Queen", "Furniture", 249.99, True, "Bedroom", "Zinus"),
]

# Create lookup dictionaries for fast access
PRODUCT_BY_ID = {p.id: p for p in PRODUCT_CATALOG}
PRODUCT_NAMES = [p.name for p in PRODUCT_CATALOG]


class WalmartMockAPI:
    """
    Mock Walmart API for product lookups and price information.
    Simulates real API behavior for the cart optimizer agent.
    """
    
    def __init__(self):
        """Initialize the mock API."""
        self.catalog = PRODUCT_CATALOG
        self.product_by_id = PRODUCT_BY_ID
        self.product_names = PRODUCT_NAMES
        print("✓ Walmart Mock API initialized")
    
    def search_product(self, query: str, threshold: int = 60) -> List[Product]:
        """
        Search for products by name using fuzzy matching.
        
        Args:
            query: Search query string.
            threshold: Minimum match score (0-100).
            
        Returns:
            List of matching products sorted by relevance.
        """
        matches = process.extract(query, self.product_names, limit=5)
        results = []
        
        for name, score in matches:
            if score >= threshold:
                for product in self.catalog:
                    if product.name == name:
                        results.append(product)
                        break
        
        return results
    
    def get_product_by_id(self, product_id: str) -> Optional[Product]:
        """
        Get a product by its ID.
        
        Args:
            product_id: Product ID.
            
        Returns:
            Product if found, None otherwise.
        """
        return self.product_by_id.get(product_id)
    
    def get_products_by_category(self, category: str) -> List[Product]:
        """
        Get all products in a category.
        
        Args:
            category: Category name.
            
        Returns:
            List of products in the category.
        """
        return [p for p in self.catalog if p.category.lower() == category.lower()]
    
    def get_price(self, product_name: str) -> Optional[float]:
        """
        Get the price of a product by name.
        
        Args:
            product_name: Product name (fuzzy matched).
            
        Returns:
            Price if found, None otherwise.
        """
        results = self.search_product(product_name, threshold=70)
        if results:
            return results[0].price
        return None
    
    def is_bnpl_eligible(self, product_name: str) -> bool:
        """
        Check if a product is eligible for BNPL.
        
        Args:
            product_name: Product name (fuzzy matched).
            
        Returns:
            True if BNPL eligible, False otherwise.
        """
        results = self.search_product(product_name, threshold=70)
        if results:
            return results[0].bnpl_eligible
        return False
    
    def get_category(self, product_name: str) -> Optional[str]:
        """
        Get the category of a product.
        
        Args:
            product_name: Product name (fuzzy matched).
            
        Returns:
            Category name if found, None otherwise.
        """
        results = self.search_product(product_name, threshold=70)
        if results:
            return results[0].category
        return None
    
    def lookup_items(self, item_names: List[str]) -> List[Dict]:
        """
        Look up multiple items and return their details.
        
        Args:
            item_names: List of item names to look up.
            
        Returns:
            List of dictionaries with item details.
        """
        results = []
        
        for name in item_names:
            matches = self.search_product(name, threshold=50)
            
            if matches:
                product = matches[0]
                results.append({
                    "query": name,
                    "matched_product": product.name,
                    "price": product.price,
                    "category": product.category,
                    "bnpl_eligible": product.bnpl_eligible,
                    "brand": product.brand,
                    "match_confidence": "high" if len(matches) == 1 else "medium"
                })
            else:
                # Return estimated values for unmatched items
                results.append({
                    "query": name,
                    "matched_product": None,
                    "price": self._estimate_price(name),
                    "category": self._guess_category(name),
                    "bnpl_eligible": False,
                    "brand": None,
                    "match_confidence": "low"
                })
        
        return results
    
    def _estimate_price(self, item_name: str) -> float:
        """Estimate price for unknown items based on name patterns."""
        name_lower = item_name.lower()
        
        # Electronics tend to be expensive
        if any(kw in name_lower for kw in ['tv', 'laptop', 'phone', 'tablet', 'console']):
            return 299.99
        elif any(kw in name_lower for kw in ['cable', 'charger', 'adapter']):
            return 14.99
        # Baby items
        elif any(kw in name_lower for kw in ['diaper', 'formula', 'wipes']):
            return 24.99
        # Food items
        elif any(kw in name_lower for kw in ['milk', 'bread', 'eggs', 'cheese']):
            return 4.99
        elif any(kw in name_lower for kw in ['meat', 'chicken', 'beef', 'fish']):
            return 8.99
        # Default
        else:
            return 9.99
    
    def _guess_category(self, item_name: str) -> str:
        """Guess category for unknown items based on name patterns."""
        name_lower = item_name.lower()
        
        category_keywords = {
            "Electronics": ['tv', 'laptop', 'phone', 'cable', 'charger', 'speaker', 'headphone'],
            "Groceries": ['milk', 'bread', 'eggs', 'cheese', 'fruit', 'vegetable', 'meat', 'chicken'],
            "Baby & Kids": ['diaper', 'baby', 'wipes', 'formula', 'toy', 'lego'],
            "Health & Beauty": ['toothpaste', 'shampoo', 'soap', 'vitamin', 'medicine'],
            "Household": ['paper', 'towel', 'cleaner', 'detergent', 'bag'],
            "Clothing": ['shirt', 'pants', 'shoes', 'jacket', 'dress'],
            "Sports": ['ball', 'yoga', 'weight', 'tent', 'fishing'],
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in name_lower for kw in keywords):
                return category
        
        return "General"
    
    def get_bnpl_eligible_items(self) -> List[Product]:
        """Get all BNPL eligible products."""
        return [p for p in self.catalog if p.bnpl_eligible]
    
    def get_category_summary(self) -> Dict[str, Dict]:
        """Get a summary of products by category."""
        summary = {}
        
        for product in self.catalog:
            if product.category not in summary:
                summary[product.category] = {
                    "count": 0,
                    "total_value": 0,
                    "bnpl_eligible_count": 0,
                    "avg_price": 0
                }
            
            summary[product.category]["count"] += 1
            summary[product.category]["total_value"] += product.price
            if product.bnpl_eligible:
                summary[product.category]["bnpl_eligible_count"] += 1
        
        # Calculate averages
        for cat in summary:
            summary[cat]["avg_price"] = round(
                summary[cat]["total_value"] / summary[cat]["count"], 2
            )
        
        return summary


# Demo/test function
if __name__ == "__main__":
    api = WalmartMockAPI()
    
    print("Testing WalmartMockAPI...")
    print("-" * 40)
    
    # Test search
    print("\nSearching for 'airpods':")
    results = api.search_product("airpods")
    for p in results:
        print(f"  {p.name}: ${p.price} (BNPL: {p.bnpl_eligible})")
    
    # Test lookup
    print("\nLooking up items:")
    items = ["diapers", "laptop", "milk", "pizza"]
    lookups = api.lookup_items(items)
    for item in lookups:
        print(f"  {item['query']} -> {item['matched_product']}: ${item['price']}")
    
    # Test category summary
    print("\nCategory Summary:")
    summary = api.get_category_summary()
    for cat, data in summary.items():
        print(f"  {cat}: {data['count']} items, avg ${data['avg_price']}")
