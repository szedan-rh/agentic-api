from openai import OpenAI

client = OpenAI()

# Test 1: Delete middle link (resp2)
print("=== Test 1: Delete middle link ===")
resp1 = client.responses.create(model="gpt-4o", input="Remember: the secret word is banana", store=True)
print(f"resp1: {resp1.id}")

resp2 = client.responses.create(model="gpt-4o", input="Acknowledge the secret word", previous_response_id=resp1.id, store=True)
print(f"resp2: {resp2.id}")

resp3 = client.responses.create(model="gpt-4o", input="Say the secret word again", previous_response_id=resp2.id, store=True)
print(f"resp3: {resp3.id} → {resp3.output_text}")

client.responses.delete(resp2.id)
print(f"Deleted resp2 (middle link)")

resp4 = client.responses.create(model="gpt-4o", input="What was the secret word?", previous_response_id=resp3.id, store=True)
print(f"resp4: {resp4.id} → {resp4.output_text}")

# Test 2: Delete the source of truth (resp1 — the one with "banana")
print("\n=== Test 2: Delete source of truth ===")
r1 = client.responses.create(model="gpt-4o", input="Remember: the secret word is mango", store=True)
print(f"r1: {r1.id}")

r2 = client.responses.create(model="gpt-4o", input="Acknowledge the secret word", previous_response_id=r1.id, store=True)
print(f"r2: {r2.id}")

r3 = client.responses.create(model="gpt-4o", input="Say the secret word again", previous_response_id=r2.id, store=True)
print(f"r3: {r3.id} → {r3.output_text}")

client.responses.delete(r1.id)
print(f"Deleted r1 (source of 'mango')")

r4 = client.responses.create(model="gpt-4o", input="What was the secret word?", previous_response_id=r3.id, store=True)
print(f"r4: {r4.id} → {r4.output_text}")
