"""
Taxonomy definition for AmazonHelp customer service interactions.
Derived from empirical analysis of customer support threads.
FROZEN IN TASK 3 - DO NOT REVISE AFTER TASK 4.
"""

INTENTS = {
    "DELIVERY_STATUS_DELAY": "Inquiries or complaints regarding missing packages, delayed shipments, tracking numbers, or estimated delivery dates.",
    "RETURN_REFUND_EXCHANGE": "Requests for returning an item, inquiring about refund status, requesting replacements or exchanges.",
    "PAYMENT_BILLING_PROMO": "Issues related to unrecognized charges, payment failures, gift cards, promotional discounts, or invoice receipts.",
    "ACCOUNT_SECURITY_ACCESS": "Issues regarding account login, password reset, account closure, 2FA, OTP, or unauthorized access.",
    "ORDER_CHANGE_CANCEL": "Requests to cancel an order, modify shipping address, change payment method, or alter order items before dispatch.",
    "PRODUCT_TECH_DIGITAL": "Inquiries or technical issues concerning Prime Video, Kindle, Fire TV, Audible, digital content, or defective product functionality.",
    "FEEDBACK_SERVICE_COMPLAINT": "General dissatisfaction with customer service agents, delivery drivers, packaging quality, or overall brand experience.",
}

INTENT_NAMES = list(INTENTS.keys())

def validate_intent(intent: str) -> bool:
    return intent in INTENTS
