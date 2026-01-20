"""
Walmart Cart Optimizer - Streamlit Application
Main entry point for the AI-powered shopping cart payment optimizer.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision_engine import CartVisionEngine, DetectedItem
from ocr_processor import ReceiptOCRProcessor, ReceiptData
from walmart_api import WalmartMockAPI
from finance_brain import FinanceBrainAgent, CartItem, create_cart_items

# Page config
st.set_page_config(
    page_title="Walmart Cart Optimizer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Cards */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Success/Warning boxes */
    .pay-now-box {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        color: white;
    }
    
    .bnpl-box {
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        color: white;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.95);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed rgba(124, 58, 237, 0.5);
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Item cards */
    .item-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        margin: 8px 0;
        border-left: 4px solid #7c3aed;
    }
    
    /* Timeline */
    .timeline-item {
        border-left: 3px solid #7c3aed;
        padding-left: 20px;
        margin-left: 10px;
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize components (cached)
@st.cache_resource
def init_components():
    """Initialize all components."""
    vision = CartVisionEngine()
    ocr = ReceiptOCRProcessor(engine="mock")  # Use mock for demo
    api = WalmartMockAPI()
    brain = FinanceBrainAgent()
    return vision, ocr, api, brain


def main():
    """Main application."""
    
    # Header
    st.title("🛒 Walmart Cart Optimizer")
    st.markdown("*AI-powered payment strategy using Buy-Now-Pay-Later*")
    
    # Initialize components
    vision, ocr, api, brain = init_components()
    
    # Sidebar - User Selection
    with st.sidebar:
        st.header("👤 User Profile")
        
        users = brain.list_users()
        user_options = {f"{u['name']} (${u['balance']:.0f})": u['id'] for u in users}
        
        selected_user_display = st.selectbox(
            "Select User",
            options=list(user_options.keys()),
            index=2  # Default to Emily (tight budget for demo)
        )
        selected_user_id = user_options[selected_user_display]
        
        # Show user financial summary
        user = brain.get_user(selected_user_id)
        finances = brain.calculate_available_funds(selected_user_id)
        
        st.markdown("---")
        st.subheader("💰 Financial Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Balance", f"${finances['current_balance']:.0f}")
        with col2:
            st.metric("Bills Due", f"${finances['total_bills']:.0f}")
        
        st.metric("Available", f"${finances['available_for_spending']:.0f}")
        
        if finances['paycheck_date']:
            st.info(f"💵 Paycheck: ${finances['paycheck_expected']:.0f} on {finances['paycheck_date'][:10]}")
        
        st.markdown("---")
        st.subheader("📋 Upcoming Bills")
        for bill in finances['upcoming_bills'][:5]:
            st.text(f"• {bill['name']}: ${bill['amount']:.0f}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📷 Cart Scanner", "🧾 Receipt Reader", "✏️ Manual Entry"])
    
    # ==================== TAB 1: Cart Scanner ====================
    with tab1:
        st.header("Upload Cart Photo")
        st.markdown("Take a photo of your shopping cart and let AI identify the items.")
        
        uploaded_cart = st.file_uploader(
            "Upload cart image",
            type=["jpg", "jpeg", "png"],
            key="cart_upload"
        )
        
        if uploaded_cart:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Your Cart")
                image = Image.open(uploaded_cart)
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("🔍 Detected Items")
                
                with st.spinner("Analyzing cart with AI..."):
                    # Save temp file for processing
                    temp_path = f"/tmp/cart_{datetime.now().timestamp()}.jpg"
                    image.save(temp_path)
                    
                    # Detect items
                    detected = vision.detect_items(temp_path)
                    summary = vision.get_cart_summary(detected)
                    
                    # Clean up
                    os.remove(temp_path)
                
                if detected:
                    for item in detected:
                        st.markdown(f"""
                        <div class="item-card">
                            <strong>{item.name}</strong><br>
                            <small>{item.category}</small> • 
                            <strong>${item.estimated_price:.2f}</strong>
                            <small style="color: #888;">({item.confidence:.0%} conf)</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.metric("Estimated Total", f"${summary['estimated_total']:.2f}")
                    
                    # Convert to CartItems for optimization
                    cart_items = []
                    for item in detected:
                        # Check BNPL eligibility via API
                        bnpl_eligible = api.is_bnpl_eligible(item.name)
                        cart_items.append(CartItem(
                            name=item.name,
                            price=item.estimated_price,
                            category=item.category,
                            bnpl_eligible=bnpl_eligible
                        ))
                    
                    st.session_state['cart_items'] = cart_items
                    st.session_state['cart_source'] = 'vision'
                else:
                    st.warning("No items detected. Try a clearer image or use Manual Entry.")
    
    # ==================== TAB 2: Receipt Reader ====================
    with tab2:
        st.header("Upload Receipt")
        st.markdown("Upload a receipt image to extract items and prices.")
        
        uploaded_receipt = st.file_uploader(
            "Upload receipt image",
            type=["jpg", "jpeg", "png"],
            key="receipt_upload"
        )
        
        if uploaded_receipt:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🧾 Receipt")
                image = Image.open(uploaded_receipt)
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("📝 Extracted Items")
                
                with st.spinner("Reading receipt with OCR..."):
                    # Save temp file
                    temp_path = f"/tmp/receipt_{datetime.now().timestamp()}.jpg"
                    image.save(temp_path)
                    
                    # Process receipt
                    receipt = ocr.parse_receipt(temp_path)
                    summary = ocr.get_receipt_summary(receipt)
                    
                    os.remove(temp_path)
                
                if receipt.items:
                    st.text(f"Store: {receipt.store_name}")
                    st.text(f"Date: {receipt.date}")
                    st.markdown("---")
                    
                    for item in receipt.items:
                        category = api.get_category(item.name) or "General"
                        st.markdown(f"""
                        <div class="item-card">
                            <strong>{item.name}</strong><br>
                            <small>{category}</small> • 
                            <strong>${item.price:.2f}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    if receipt.subtotal:
                        st.text(f"Subtotal: ${receipt.subtotal:.2f}")
                    if receipt.tax:
                        st.text(f"Tax: ${receipt.tax:.2f}")
                    if receipt.total:
                        st.metric("Total", f"${receipt.total:.2f}")
                    
                    # Convert to CartItems
                    cart_items = []
                    for item in receipt.items:
                        category = api.get_category(item.name) or "General"
                        bnpl_eligible = api.is_bnpl_eligible(item.name)
                        cart_items.append(CartItem(
                            name=item.name,
                            price=item.price,
                            category=category,
                            bnpl_eligible=bnpl_eligible
                        ))
                    
                    st.session_state['cart_items'] = cart_items
                    st.session_state['cart_source'] = 'receipt'
    
    # ==================== TAB 3: Manual Entry ====================
    with tab3:
        st.header("Manual Entry")
        st.markdown("Add items manually to your cart.")
        
        # Initialize manual cart
        if 'manual_cart' not in st.session_state:
            st.session_state['manual_cart'] = []
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            item_name = st.text_input("Item Name", placeholder="e.g., Apple AirPods")
        with col2:
            item_price = st.number_input("Price ($)", min_value=0.01, value=49.99, step=0.01)
        with col3:
            categories = ["Electronics", "Groceries", "Baby & Kids", "Health & Beauty", 
                         "Household", "Clothing", "Sports", "Furniture"]
            item_category = st.selectbox("Category", categories)
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add"):
                if item_name:
                    st.session_state['manual_cart'].append({
                        "name": item_name,
                        "price": item_price,
                        "category": item_category
                    })
        
        # Quick add common items
        st.markdown("**Quick Add:**")
        quick_items = [
            ("🍌 Bananas", 1.49, "Groceries"),
            ("🥛 Milk", 3.89, "Groceries"),
            ("👶 Diapers", 24.99, "Baby & Kids"),
            ("🎧 AirPods", 149.99, "Electronics"),
            ("📺 TV", 399.99, "Electronics"),
            ("🧥 Jacket", 49.99, "Clothing"),
        ]
        
        cols = st.columns(6)
        for i, (label, price, cat) in enumerate(quick_items):
            with cols[i]:
                if st.button(label, key=f"quick_{i}"):
                    st.session_state['manual_cart'].append({
                        "name": label.split(" ", 1)[1],
                        "price": price,
                        "category": cat
                    })
        
        # Show current cart
        if st.session_state['manual_cart']:
            st.markdown("---")
            st.subheader("🛒 Your Cart")
            
            total = 0
            for i, item in enumerate(st.session_state['manual_cart']):
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.text(f"{item['name']} ({item['category']})")
                with col2:
                    st.text(f"${item['price']:.2f}")
                with col3:
                    if st.button("❌", key=f"del_{i}"):
                        st.session_state['manual_cart'].pop(i)
                        st.rerun()
                total += item['price']
            
            st.markdown("---")
            st.metric("Cart Total", f"${total:.2f}")
            
            # Convert to CartItems
            cart_items = create_cart_items(st.session_state['manual_cart'])
            for item in cart_items:
                item.bnpl_eligible = api.is_bnpl_eligible(item.name)
            
            st.session_state['cart_items'] = cart_items
            st.session_state['cart_source'] = 'manual'
    
    # ==================== OPTIMIZATION SECTION ====================
    st.markdown("---")
    
    if 'cart_items' in st.session_state and st.session_state['cart_items']:
        st.header("💡 Payment Optimization")
        
        if st.button("🔮 Optimize My Payments", type="primary", use_container_width=True):
            with st.spinner("Analyzing your finances and cart..."):
                result = brain.optimize_cart(selected_user_id, st.session_state['cart_items'])
            
            st.session_state['optimization_result'] = result
        
        if 'optimization_result' in st.session_state:
            result = st.session_state['optimization_result']
            
            # Summary
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(0, 212, 255, 0.2)); 
                        padding: 25px; border-radius: 15px; margin: 20px 0;">
                {result.summary.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            # Warnings
            for warning in result.warnings:
                st.markdown(f"""
                <div class="warning-box">
                    {warning}
                </div>
                """, unsafe_allow_html=True)
            
            # Payment breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="pay-now-box">
                    <h3 style="color: white; margin: 0;">💳 Pay Now</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if result.pay_now_items:
                    for item in result.pay_now_items:
                        st.markdown(f"• **{item.name}** - ${item.price:.2f}")
                    st.metric("Pay Now Total", f"${result.pay_now_total:.2f}")
                else:
                    st.info("No items to pay now")
            
            with col2:
                st.markdown("""
                <div class="bnpl-box">
                    <h3 style="color: white; margin: 0;">📅 Buy Now, Pay Later</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if result.bnpl_items:
                    for item in result.bnpl_items:
                        st.markdown(f"• **{item.name}** - ${item.price:.2f}")
                    st.metric("BNPL Total", f"${result.bnpl_total:.2f}")
                    st.caption(f"4 payments of ${result.monthly_bnpl_payment:.2f} every 2 weeks")
                else:
                    st.info("No items for BNPL")
            
            # Payment calendar
            st.markdown("---")
            st.subheader("📅 Payment Calendar")
            
            calendar = brain.get_payment_calendar(result, selected_user_id)
            
            for event in calendar[:8]:
                icon = "💳" if event['type'] == 'PAYMENT' else "📅" if event['type'] == 'BNPL_PAYMENT' else "📄" if event['type'] == 'BILL' else "💰"
                color = "#059669" if event['type'] == 'INCOME' else "#dc2626" if event['type'] == 'BILL' else "#7c3aed"
                
                st.markdown(f"""
                <div class="timeline-item">
                    <strong>{event['date']}</strong> {icon}<br>
                    {event['description']}<br>
                    <span style="color: {color}; font-weight: bold;">
                        {'+'if event['amount'] > 0 else ''}${event['amount']:.2f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # Final balance
            st.markdown("---")
            balance_color = "#059669" if result.projected_balance > 100 else "#dc2626"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h2>Projected Balance After Purchase</h2>
                <h1 style="color: {balance_color}; font-size: 48px;">${result.projected_balance:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👆 Upload a cart photo, receipt, or add items manually to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
        <p>🤖 Powered by PyTorch, OCR, and AI Agent Technology</p>
        <p><small>Demo project showcasing Computer Vision + BNPL Financial Logic</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
