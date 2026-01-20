"""
Finance Brain Module - The Core BNPL Agent
Analyzes user finances and proposes optimal payment strategies.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

# Optional LangChain integration
try:
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@dataclass
class CartItem:
    """Represents an item in the shopping cart."""
    name: str
    price: float
    category: str
    bnpl_eligible: bool = True
    is_essential: bool = False


@dataclass
class PaymentPlan:
    """Represents a BNPL payment plan."""
    installments: int
    interval_weeks: int = 2
    interval_months: int = 0
    fee_percent: float = 0.0


@dataclass
class PaymentRecommendation:
    """Recommendation for a single item."""
    item: CartItem
    strategy: str  # "PAY_NOW" or "BNPL"
    reason: str
    payment_plan: Optional[PaymentPlan] = None
    installment_amount: float = 0.0
    payment_dates: List[str] = field(default_factory=list)


@dataclass
class CartOptimization:
    """Complete cart optimization result."""
    pay_now_items: List[CartItem]
    bnpl_items: List[CartItem]
    pay_now_total: float
    bnpl_total: float
    monthly_bnpl_payment: float
    recommendations: List[PaymentRecommendation]
    summary: str
    warnings: List[str]
    projected_balance: float


class FinanceBrainAgent:
    """
    The intelligent agent that analyzes financial data and proposes 
    optimal payment splits using BNPL logic.
    """
    
    # Categories considered essential (pay now)
    ESSENTIAL_CATEGORIES = [
        "Groceries", "Baby & Kids", "Health & Beauty", "Medicine"
    ]
    
    # Minimum purchase amount for BNPL
    BNPL_MIN_AMOUNT = 35.0
    
    # Maximum BNPL amount
    BNPL_MAX_AMOUNT = 2000.0
    
    def __init__(self, user_db_path: Optional[str] = None, use_llm: bool = False):
        """
        Initialize the Finance Brain agent.
        
        Args:
            user_db_path: Path to the mock user database JSON.
            use_llm: Whether to use LLM for natural language explanations.
        """
        self.users = {}
        self.bnpl_config = {}
        self.use_llm = use_llm and LANGCHAIN_AVAILABLE
        self.llm_chain = None
        
        # Load user database
        if user_db_path and os.path.exists(user_db_path):
            self._load_user_db(user_db_path)
        else:
            # Try default path
            default_path = Path(__file__).parent.parent / "data" / "mock_user_db.json"
            if default_path.exists():
                self._load_user_db(str(default_path))
        
        # Initialize LLM if available and requested
        if self.use_llm:
            self._init_llm()
        
        print("✓ Finance Brain Agent initialized")
    
    def _load_user_db(self, path: str):
        """Load user database from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
            
        for user in data.get("users", []):
            self.users[user["id"]] = user
        
        self.bnpl_config = data.get("bnpl_config", {})
        print(f"  Loaded {len(self.users)} user profiles")
    
    def _init_llm(self):
        """Initialize LangChain LLM for natural language explanations."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("  ⚠ OPENAI_API_KEY not set, using rule-based explanations")
                self.use_llm = False
                return
            
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                api_key=api_key
            )
            
            prompt = PromptTemplate(
                input_variables=["context", "items", "recommendation"],
                template="""You are a friendly financial advisor helping a customer optimize their Walmart shopping payment strategy.

Context about the customer:
{context}

Shopping cart items:
{items}

Payment recommendation:
{recommendation}

Provide a brief, friendly explanation (2-3 sentences) of why this payment strategy makes sense for this customer. Focus on how it helps their cash flow and financial health. Be conversational but professional."""
            )
            
            self.llm_chain = LLMChain(llm=llm, prompt=prompt)
            print("  ✓ LLM integration enabled")
        except Exception as e:
            print(f"  ⚠ LLM init failed: {e}, using rule-based explanations")
            self.use_llm = False
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID."""
        return self.users.get(user_id)
    
    def list_users(self) -> List[Dict]:
        """List all available user profiles."""
        return [
            {"id": u["id"], "name": u["name"], "balance": u["bank_balance"]}
            for u in self.users.values()
        ]
    
    def calculate_available_funds(
        self, 
        user_id: str, 
        days_ahead: int = 30
    ) -> Dict:
        """
        Calculate user's available funds after accounting for upcoming bills.
        
        Args:
            user_id: User ID.
            days_ahead: Number of days to look ahead.
            
        Returns:
            Dictionary with financial analysis.
        """
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        today = datetime.now()
        cutoff = today + timedelta(days=days_ahead)
        
        # Current balance
        balance = user["bank_balance"]
        
        # Calculate upcoming bills within the period
        upcoming_bills = []
        total_bills = 0
        
        for bill in user.get("upcoming_bills", []):
            due_date = parse_date(bill["due_date"])
            if today <= due_date <= cutoff:
                upcoming_bills.append({
                    "name": bill["name"],
                    "amount": bill["amount"],
                    "due_date": bill["due_date"],
                    "days_until": (due_date - today).days
                })
                total_bills += bill["amount"]
        
        # Check for incoming paycheck
        paycheck = user.get("next_paycheck", {})
        paycheck_amount = 0
        paycheck_date = None
        
        if paycheck:
            paycheck_date = parse_date(paycheck["date"])
            if today <= paycheck_date <= cutoff:
                paycheck_amount = paycheck["amount"]
        
        # Calculate projected balance
        projected_balance = balance - total_bills + paycheck_amount
        
        # Available for discretionary spending (keep 10% buffer)
        buffer = balance * 0.10
        available_for_spending = max(0, balance - total_bills - buffer)
        
        # Safe BNPL installment amount (should fit within available funds)
        safe_installment = available_for_spending * 0.25  # Max 25% of available
        
        return {
            "current_balance": balance,
            "upcoming_bills": upcoming_bills,
            "total_bills": total_bills,
            "paycheck_expected": paycheck_amount,
            "paycheck_date": paycheck_date.isoformat() if paycheck_date else None,
            "projected_balance": round(projected_balance, 2),
            "available_for_spending": round(available_for_spending, 2),
            "safe_bnpl_installment": round(safe_installment, 2),
            "credit_tier": user.get("credit_tier", "unknown"),
            "bnpl_eligible": user.get("bnpl_eligible", False)
        }
    
    def classify_items(self, items: List[CartItem]) -> Tuple[List[CartItem], List[CartItem]]:
        """
        Classify items into essential (pay now) and discretionary (BNPL candidate).
        
        Args:
            items: List of cart items.
            
        Returns:
            Tuple of (essential items, discretionary items).
        """
        essential = []
        discretionary = []
        
        for item in items:
            # Mark as essential based on category
            is_essential = item.category in self.ESSENTIAL_CATEGORIES
            item.is_essential = is_essential
            
            if is_essential:
                essential.append(item)
            elif item.bnpl_eligible and item.price >= self.BNPL_MIN_AMOUNT:
                discretionary.append(item)
            else:
                # Low-cost non-essential items should be paid now
                essential.append(item)
        
        return essential, discretionary
    
    def calculate_bnpl_plan(
        self, 
        amount: float, 
        installments: int = 4
    ) -> Dict:
        """
        Calculate BNPL payment plan details.
        
        Args:
            amount: Total amount to finance.
            installments: Number of payments.
            
        Returns:
            Payment plan details.
        """
        today = datetime.now()
        
        # Standard 4-pay plan: every 2 weeks
        installment_amount = round(amount / installments, 2)
        
        # Adjust for rounding (last payment covers remainder)
        payments = [installment_amount] * (installments - 1)
        payments.append(round(amount - sum(payments), 2))
        
        # Payment dates (every 2 weeks)
        dates = []
        for i in range(installments):
            payment_date = today + timedelta(weeks=2 * i)
            dates.append(payment_date.strftime("%Y-%m-%d"))
        
        return {
            "total_amount": amount,
            "installments": installments,
            "installment_amount": installment_amount,
            "payments": payments,
            "payment_dates": dates,
            "fee": 0.0,  # OnePay typically 0% interest
            "total_cost": amount  # No additional cost
        }
    
    def optimize_cart(
        self, 
        user_id: str, 
        items: List[CartItem]
    ) -> CartOptimization:
        """
        Optimize a shopping cart with BNPL recommendations.
        
        Args:
            user_id: User ID for financial analysis.
            items: List of items in the cart.
            
        Returns:
            CartOptimization with recommendations.
        """
        # Get user's financial situation
        finances = self.calculate_available_funds(user_id)
        user = self.get_user(user_id)
        
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        # Classify items
        essential, discretionary = self.classify_items(items)
        
        # Calculate totals
        essential_total = sum(item.price for item in essential)
        discretionary_total = sum(item.price for item in discretionary)
        cart_total = essential_total + discretionary_total
        
        # Determine what can be paid now
        available = finances["available_for_spending"]
        safe_installment = finances["safe_bnpl_installment"]
        
        recommendations = []
        warnings = []
        pay_now_items = []
        bnpl_items = []
        
        # Essential items should always be paid now
        for item in essential:
            rec = PaymentRecommendation(
                item=item,
                strategy="PAY_NOW",
                reason=f"{item.category} items are essential and should be paid immediately."
            )
            recommendations.append(rec)
            pay_now_items.append(item)
        
        # Analyze discretionary items for BNPL
        for item in discretionary:
            if not finances["bnpl_eligible"]:
                # User not eligible for BNPL
                rec = PaymentRecommendation(
                    item=item,
                    strategy="PAY_NOW",
                    reason="BNPL not available for your account."
                )
                pay_now_items.append(item)
            elif item.price > self.BNPL_MAX_AMOUNT:
                # Item exceeds BNPL limit
                rec = PaymentRecommendation(
                    item=item,
                    strategy="PAY_NOW",
                    reason=f"Item exceeds BNPL limit of ${self.BNPL_MAX_AMOUNT}."
                )
                pay_now_items.append(item)
                warnings.append(f"{item.name} exceeds BNPL maximum amount.")
            elif essential_total > available:
                # Can't afford essentials, definitely use BNPL
                plan = self.calculate_bnpl_plan(item.price)
                rec = PaymentRecommendation(
                    item=item,
                    strategy="BNPL",
                    reason="Your essential purchases already exceed available funds. "
                           "BNPL helps preserve cash for necessities.",
                    payment_plan=PaymentPlan(installments=4),
                    installment_amount=plan["installment_amount"],
                    payment_dates=plan["payment_dates"]
                )
                bnpl_items.append(item)
                warnings.append("Budget is tight! Consider if all items are necessary.")
            elif essential_total + item.price > available:
                # This item would exceed budget
                plan = self.calculate_bnpl_plan(item.price)
                rec = PaymentRecommendation(
                    item=item,
                    strategy="BNPL",
                    reason=f"Paying now would leave you with only "
                           f"${available - essential_total - item.price:.2f}. "
                           f"Use BNPL to maintain a safety buffer.",
                    payment_plan=PaymentPlan(installments=4),
                    installment_amount=plan["installment_amount"],
                    payment_dates=plan["payment_dates"]
                )
                bnpl_items.append(item)
            elif plan_installment := (item.price / 4) <= safe_installment:
                # BNPL installment is affordable
                plan = self.calculate_bnpl_plan(item.price)
                rec = PaymentRecommendation(
                    item=item,
                    strategy="BNPL",
                    reason=f"At ${plan['installment_amount']:.2f} per payment, "
                           f"this fits comfortably in your budget while preserving "
                           f"cash for unexpected expenses.",
                    payment_plan=PaymentPlan(installments=4),
                    installment_amount=plan["installment_amount"],
                    payment_dates=plan["payment_dates"]
                )
                bnpl_items.append(item)
            else:
                # User can afford to pay now
                rec = PaymentRecommendation(
                    item=item,
                    strategy="PAY_NOW",
                    reason="You have sufficient funds. Paying now avoids future obligations."
                )
                pay_now_items.append(item)
            
            recommendations.append(rec)
        
        # Calculate final totals
        pay_now_total = sum(item.price for item in pay_now_items)
        bnpl_total = sum(item.price for item in bnpl_items)
        monthly_bnpl = bnpl_total / 4 if bnpl_items else 0
        
        # Check if the first payment is affordable
        first_payment = pay_now_total + monthly_bnpl
        projected = finances["current_balance"] - first_payment
        
        if projected < 0:
            warnings.append(
                f"⚠ Warning: This purchase would overdraw your account by ${abs(projected):.2f}!"
            )
        elif projected < 100:
            warnings.append(
                f"⚠ Caution: This would leave only ${projected:.2f} in your account."
            )
        
        # Generate summary
        summary = self._generate_summary(
            user, finances, pay_now_items, bnpl_items, 
            pay_now_total, bnpl_total, projected
        )
        
        return CartOptimization(
            pay_now_items=pay_now_items,
            bnpl_items=bnpl_items,
            pay_now_total=round(pay_now_total, 2),
            bnpl_total=round(bnpl_total, 2),
            monthly_bnpl_payment=round(monthly_bnpl, 2),
            recommendations=recommendations,
            summary=summary,
            warnings=warnings,
            projected_balance=round(projected, 2)
        )
    
    def _generate_summary(
        self, 
        user: Dict,
        finances: Dict,
        pay_now_items: List[CartItem],
        bnpl_items: List[CartItem],
        pay_now_total: float,
        bnpl_total: float,
        projected_balance: float
    ) -> str:
        """Generate a natural language summary of the recommendation."""
        
        # Try LLM if available
        if self.use_llm and self.llm_chain:
            try:
                context = f"""
                Name: {user['name']}
                Current Balance: ${finances['current_balance']:.2f}
                Upcoming Bills: ${finances['total_bills']:.2f}
                Next Paycheck: ${finances['paycheck_expected']:.2f} on {finances['paycheck_date']}
                """
                
                items_str = "\n".join([
                    f"- {item.name} (${item.price:.2f}, {item.category})"
                    for item in pay_now_items + bnpl_items
                ])
                
                rec_str = f"""
                Pay Now: {len(pay_now_items)} items totaling ${pay_now_total:.2f}
                BNPL: {len(bnpl_items)} items totaling ${bnpl_total:.2f}
                (4 payments of ${bnpl_total/4:.2f} every 2 weeks)
                """
                
                result = self.llm_chain.run(
                    context=context,
                    items=items_str,
                    recommendation=rec_str
                )
                return result.strip()
            except Exception as e:
                print(f"LLM generation failed: {e}")
        
        # Rule-based summary
        if not bnpl_items:
            return (
                f"✅ Great news! You can comfortably pay ${pay_now_total:.2f} for all items today.\n"
                f"Your projected balance after this purchase: ${projected_balance:.2f}"
            )
        
        summary_parts = [
            f"💡 **Smart Payment Strategy for {user['name']}**\n",
            f"**Today's Payment:** ${pay_now_total:.2f} for essentials "
            f"({len(pay_now_items)} items)\n",
            f"**BNPL Items:** ${bnpl_total:.2f} split into 4 payments of "
            f"${bnpl_total/4:.2f} every 2 weeks\n",
        ]
        
        # Add specific item callouts
        if bnpl_items:
            bnpl_names = ", ".join(item.name for item in bnpl_items[:3])
            summary_parts.append(f"\n📦 **Financing:** {bnpl_names}\n")
        
        # Add financial reasoning
        if finances["paycheck_expected"] > 0:
            summary_parts.append(
                f"\n💰 With your next paycheck of ${finances['paycheck_expected']:.2f} "
                f"coming on {finances['paycheck_date']}, this strategy ensures you "
                f"have enough for rent and bills while getting everything you need today."
            )
        
        return "".join(summary_parts)
    
    def get_payment_calendar(
        self, 
        optimization: CartOptimization,
        user_id: str
    ) -> List[Dict]:
        """
        Generate a payment calendar showing when payments are due.
        
        Args:
            optimization: Cart optimization result.
            user_id: User ID.
            
        Returns:
            List of calendar events.
        """
        user = self.get_user(user_id)
        finances = self.calculate_available_funds(user_id)
        
        events = []
        
        # Today: immediate payment
        events.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "type": "PAYMENT",
            "description": "Cart Purchase - Pay Now Items",
            "amount": -optimization.pay_now_total,
            "balance_after": finances["current_balance"] - optimization.pay_now_total
        })
        
        # BNPL payments
        if optimization.bnpl_items:
            for i in range(4):
                payment_date = datetime.now() + timedelta(weeks=2 * i)
                amount = optimization.bnpl_total / 4
                events.append({
                    "date": payment_date.strftime("%Y-%m-%d"),
                    "type": "BNPL_PAYMENT",
                    "description": f"BNPL Payment {i+1}/4",
                    "amount": -amount,
                    "items": [item.name for item in optimization.bnpl_items]
                })
        
        # Add upcoming bills
        for bill in finances["upcoming_bills"]:
            events.append({
                "date": bill["due_date"],
                "type": "BILL",
                "description": bill["name"],
                "amount": -bill["amount"]
            })
        
        # Add paycheck
        if finances["paycheck_date"]:
            events.append({
                "date": finances["paycheck_date"],
                "type": "INCOME",
                "description": "Paycheck",
                "amount": finances["paycheck_expected"]
            })
        
        # Sort by date
        events.sort(key=lambda x: x["date"])
        
        return events


# Helper function to create CartItem from various sources
def create_cart_items(items_data: List[Dict]) -> List[CartItem]:
    """
    Create CartItem objects from raw data.
    
    Args:
        items_data: List of dictionaries with item info.
        
    Returns:
        List of CartItem objects.
    """
    cart_items = []
    
    for item in items_data:
        cart_items.append(CartItem(
            name=item.get("name", "Unknown Item"),
            price=item.get("price", 0.0),
            category=item.get("category", "General"),
            bnpl_eligible=item.get("bnpl_eligible", True)
        ))
    
    return cart_items


# Demo/test function
if __name__ == "__main__":
    # Initialize agent
    agent = FinanceBrainAgent()
    
    print("\nTesting FinanceBrainAgent...")
    print("-" * 50)
    
    # List users
    print("\nAvailable Users:")
    for user in agent.list_users():
        print(f"  {user['id']}: {user['name']} (${user['balance']:.2f})")
    
    # Test with Emily (tight budget)
    print("\n" + "=" * 50)
    print("Testing with Emily Rodriguez (tight budget)")
    print("=" * 50)
    
    finances = agent.calculate_available_funds("user_003")
    print(f"\nFinancial Situation:")
    print(f"  Current Balance: ${finances['current_balance']:.2f}")
    print(f"  Upcoming Bills: ${finances['total_bills']:.2f}")
    print(f"  Available for Spending: ${finances['available_for_spending']:.2f}")
    
    # Sample cart
    cart = create_cart_items([
        {"name": "Groceries Bundle", "price": 52.00, "category": "Groceries"},
        {"name": "Diapers", "price": 24.99, "category": "Baby & Kids"},
        {"name": "Apple AirPods", "price": 149.99, "category": "Electronics"},
        {"name": "Winter Jacket", "price": 49.99, "category": "Clothing"},
    ])
    
    print(f"\nCart Total: ${sum(item.price for item in cart):.2f}")
    
    # Optimize
    result = agent.optimize_cart("user_003", cart)
    
    print(f"\n{result.summary}")
    
    print(f"\n📋 Recommendations:")
    for rec in result.recommendations:
        print(f"  [{rec.strategy}] {rec.item.name} (${rec.item.price:.2f})")
        print(f"      → {rec.reason}")
        if rec.payment_dates:
            print(f"      Payment dates: {', '.join(rec.payment_dates[:2])}...")
    
    if result.warnings:
        print(f"\n⚠️ Warnings:")
        for warning in result.warnings:
            print(f"  {warning}")
