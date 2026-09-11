"""
Grounded reply generator using Gemini Flash.
Takes the customer thread + intent + top-k retrieved examples and generates a reply
grounded strictly in the retrieved evidence.
"""
from gemini_client import gemini_generate

GENERATOR_PROMPT_TEMPLATE = """You are a customer support agent for AmazonHelp on Twitter.
Generate a helpful, empathetic reply to the customer's message.

CRITICAL RULES:
1. ONLY use information present in the retrieved historical examples below.
2. Do NOT state policies, promises, refunds, timelines, or compensation not evidenced in the examples.
3. If the examples don't contain enough information, direct the customer to contact support via phone/chat.
4. Keep the reply concise (1-3 sentences), professional, and empathetic.
5. Sign off with ^AI at the end.

Customer's message: "{inbound_text}"
Detected intent: {intent}

Retrieved similar resolved cases:
{retrieved_context}

Generate ONLY the reply text, nothing else:"""

def generate_reply(inbound_text: str, intent: str, retrieved_examples: list) -> str:
    """
    Generate a grounded reply using retrieved historical examples as evidence.
    """
    context_parts = []
    for i, ex in enumerate(retrieved_examples):
        context_parts.append(
            f"Example {i+1} (similarity: {ex['similarity']:.2f}):\n"
            f"  Customer: {ex['inbound_text']}\n"
            f"  Agent Reply: {ex['support_reply']}"
        )
    
    retrieved_context = "\n\n".join(context_parts) if context_parts else "No similar cases found."
    
    prompt = GENERATOR_PROMPT_TEMPLATE.format(
        inbound_text=inbound_text,
        intent=intent,
        retrieved_context=retrieved_context
    )
    
    reply = gemini_generate(prompt, temperature=0.3, max_tokens=200)
    
    if not reply:
        # Template fallback if API fails
        reply = fallback_reply(intent)
    
    return reply

def fallback_reply(intent: str) -> str:
    """Template-based fallback reply per intent."""
    templates = {
        "DELIVERY_STATUS_DELAY": "We're sorry about the delay with your order. Please reach us via phone or chat here so we can look into it: https://t.co/hApLpMlfHN ^AI",
        "RETURN_REFUND_EXCHANGE": "We'd like to help with your return/refund. Please contact us via phone or chat here: https://t.co/hApLpMlfHN ^AI",
        "PAYMENT_BILLING_PROMO": "We understand billing concerns are urgent. Please reach us via phone or chat for account-specific help: https://t.co/hApLpMlfHN ^AI",
        "ACCOUNT_SECURITY_ACCESS": "Account security is our priority. Please contact us immediately via: https://t.co/hApLpMlfHN ^AI",
        "ORDER_CHANGE_CANCEL": "We'd like to help with your order change. Please reach us via phone or chat here: https://t.co/hApLpMlfHN ^AI",
        "PRODUCT_TECH_DIGITAL": "We'd love to help troubleshoot this. Please contact our tech support via: https://t.co/hApLpMlfHN ^AI",
        "FEEDBACK_SERVICE_COMPLAINT": "We're sorry to hear about your experience. Your feedback matters. Please share more details via: https://t.co/hApLpMlfHN ^AI",
    }
    return templates.get(intent, "We'd like to help! Please contact us via: https://t.co/hApLpMlfHN ^AI")

if __name__ == "__main__":
    mock_examples = [
        {"inbound_text": "My package hasn't arrived yet", "support_reply": "Sorry about that! Let us look into it. Can you DM us your order number?", "similarity": 0.85},
        {"inbound_text": "Where is my delivery?", "support_reply": "We'd like to help track this. Please reach us via phone or chat: https://t.co/hApLpMlfHN ^TN", "similarity": 0.78}
    ]
    reply = generate_reply("My order was supposed to arrive 3 days ago but tracking shows no update!", "DELIVERY_STATUS_DELAY", mock_examples)
    print("Generated reply:", reply)
