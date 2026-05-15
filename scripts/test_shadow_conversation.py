from openai import OpenAI

client = OpenAI()

# Create a 3-response chain
resp1 = client.responses.create(model="gpt-4o", input="Remember: the secret word is banana", store=True)
print(f"resp1: {resp1.id}")

resp2 = client.responses.create(model="gpt-4o", input="Acknowledge the secret word", previous_response_id=resp1.id, store=True)
print(f"resp2: {resp2.id}")

resp3 = client.responses.create(model="gpt-4o", input="Say the secret word again", previous_response_id=resp2.id, store=True)
print(f"resp3: {resp3.id} → {resp3.output_text}")

# Delete the middle link
client.responses.delete(resp2.id)
print(f"Deleted resp2: {resp2.id}")

# Try to continue from resp3 — does it still work?
try:
    resp4 = client.responses.create(model="gpt-4o", input="What was the secret word?", previous_response_id=resp3.id, store=True)
    print(f"resp4: {resp4.id} → {resp4.output_text}")
    print("Chain survived deletion → likely shadow conversation")
except Exception as e:
    print(f"Chain broke → likely walking the chain: {e}")

# Also dump resp1 for any hidden fields
print(f"\nFull resp1 dump keys: {list(resp1.model_dump().keys())}")
