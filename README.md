# 🛒 Walmart Cart Optimizer Agent

An AI-powered agent that analyzes shopping cart photos or receipts and optimizes payment strategies using Buy-Now-Pay-Later (BNPL) logic based on the user's financial situation.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project demonstrates the integration of:
- **Computer Vision** (YOLOv8) for shopping cart item detection
- **OCR** (Tesseract/EasyOCR) for receipt text extraction
- **Agentic AI Logic** for financial decision-making
- **BNPL Optimization** to help users manage cash flow

### The Problem

Many shoppers struggle to balance immediate purchases with upcoming bills. This agent helps by:
1. Identifying items in a shopping cart photo
2. Analyzing the user's financial situation (balance, bills, payday)
3. Recommending which items to pay now vs. finance with BNPL

### Example Recommendation

> *"Pay for the groceries ($50) now, but put the Electronics ($200) on a 4-payment plan to ensure you have enough for rent on the 1st of the month."*

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Cart Scanner │  │Receipt Reader│  │ Manual Entry │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │              │
│         ▼                 ▼                  ▼              │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │Vision Engine │  │OCR Processor │                        │
│  │   (YOLO)     │  │ (Tesseract)  │                        │
│  └──────┬───────┘  └──────┬───────┘                        │
│         │                 │                                 │
│         └────────┬────────┘                                 │
│                  ▼                                          │
│         ┌──────────────┐                                    │
│         │ Walmart API  │  ◄── Price Lookup & Categories     │
│         └──────┬───────┘                                    │
│                │                                            │
│                ▼                                            │
│         ┌──────────────┐                                    │
│         │Finance Brain │  ◄── BNPL Agent Logic              │
│         │   (Agent)    │                                    │
│         └──────┬───────┘                                    │
│                │                                            │
│                ▼                                            │
│     ┌────────────────────┐                                  │
│     │ Payment Strategy   │                                  │
│     │ Recommendation     │                                  │
│     └────────────────────┘                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
walmart-cart-agent/
├── data/
│   ├── sample_carts/         # Test cart images
│   ├── receipts/             # Sample receipt images
│   └── mock_user_db.json     # User financial profiles
│
├── src/
│   ├── __init__.py
│   ├── vision_engine.py      # YOLOv8 item detection
│   ├── ocr_processor.py      # Receipt text extraction
│   ├── walmart_api.py        # Mock product database
│   └── finance_brain.py      # BNPL optimization agent
│
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- (Optional) Tesseract OCR installed on system

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/walmart-cart-agent.git
cd walmart-cart-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
# Start the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 💡 Features

### 1. Cart Scanner (Computer Vision)
Upload a photo of your shopping cart and the AI will:
- Detect items using YOLOv8
- Classify items into categories (Electronics, Groceries, etc.)
- Estimate prices based on detected items

### 2. Receipt Reader (OCR)
Upload a receipt image to:
- Extract item names and prices
- Parse store info and totals
- Categorize items automatically

### 3. Manual Entry
Add items directly to your virtual cart with:
- Quick-add buttons for common items
- Category selection
- Custom pricing

### 4. Payment Optimization
The Finance Brain agent analyzes:
- Your current bank balance
- Upcoming bills (rent, utilities, etc.)
- Next paycheck date and amount
- Item categories (essential vs. discretionary)

Then recommends:
- **Pay Now**: Essential items like groceries, baby supplies
- **BNPL**: Large discretionary items that can be financed

## 🧠 Technical Deep Dive

### Vision Engine (`vision_engine.py`)

Uses YOLOv8 pre-trained on COCO dataset with a custom mapping to Walmart items:

```python
# Example detection
engine = CartVisionEngine()
items = engine.detect_items("cart_photo.jpg")

for item in items:
    print(f"{item.name}: ${item.estimated_price} ({item.category})")
```

### OCR Processor (`ocr_processor.py`)

Supports both Tesseract and EasyOCR with automatic fallback:

```python
processor = ReceiptOCRProcessor(engine="auto")
receipt = processor.parse_receipt("receipt.jpg")

for item in receipt.items:
    print(f"{item.name}: ${item.price}")
```

### Finance Brain (`finance_brain.py`)

The core agent logic for payment optimization:

```python
agent = FinanceBrainAgent()

# Get user's financial situation
finances = agent.calculate_available_funds("user_001")

# Optimize cart payment
result = agent.optimize_cart("user_001", cart_items)

print(result.summary)
# "Pay $52 for groceries now, put $200 AirPods on 4-pay BNPL"
```

### BNPL Logic

The agent uses these rules:

1. **Essential Categories** (always pay now):
   - Groceries, Baby & Kids, Health & Beauty, Medicine

2. **BNPL Candidates**:
   - Items $35+ that are discretionary
   - Electronics, Clothing, Furniture, etc.

3. **Financial Checks**:
   - Ensures essentials fit in available funds
   - Recommends BNPL when paying now would stress budget
   - Checks that installments fit in bi-weekly cash flow

## 📊 Sample User Profiles

The mock database includes diverse financial profiles:

| User | Balance | Paycheck | Situation |
|------|---------|----------|-----------|
| Sarah | $1,250 | $2,800/mo | Moderate budget |
| Marcus | $3,500 | $4,200/mo | Comfortable |
| Emily | $420 | $1,600/mo | Tight budget |
| David | $8,200 | $5,500/mo | High earner |

## 🔮 Future Enhancements

- [ ] Real Walmart API integration
- [ ] Custom-trained YOLO model for retail items
- [ ] User authentication and persistent profiles
- [ ] SMS/Email payment reminders
- [ ] Credit score impact analysis
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Streamlit for the amazing web framework
- LangChain for agent abstractions

---

Built with ❤️ as a demonstration of AI/ML + Financial Technology integration.
