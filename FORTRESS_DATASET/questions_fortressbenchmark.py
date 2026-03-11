import json
from openai import OpenAI

# ==========================================
# 1. SETUP YOUR NOVITA AI CONNECTION
# ==========================================
client = OpenAI(
    api_key="APIKEY", 
    base_url="https://router.huggingface.co/v1"
)

NOVITA_MODEL = "deepseek-ai/DeepSeek-V3.2:novita"
SYSTEM_PROMPT = """SYSTEM PROMPT (jailbrek injection or normal alignment)"""

# ==========================================
# 2. LOAD THE LOCAL DATASET
# ==========================================
print("Loading local rows.json file...")

# Load the file you just downloaded instead of requesting it from the internet again
with open("rows.json", "r", encoding="utf-16") as f:
    data = json.load(f)

results = []
total_rows = len(data['rows'])

# ==========================================
# 3. LOOP THROUGH AND TEST YOUR MODEL
# ==========================================
print(f"Testing model against {total_rows} prompts...")

for i, row_data in enumerate(data['rows']):
    # Drill down into the 'row' object based on the JSON structure
    row = row_data['row']
    
    # Extract the correct keys based on your JSON file
    prompt_id = row['ID']
    adversarial_prompt = row['adversarial_prompt']
    benign_prompt = row['benign_prompt']
    risk_domain = row['risk_domain']
    risk_subdomain = row['risk_subdomain']
    rubric = row['rubric']
    
    print(f"[{i+1}/{total_rows}] Testing ID {prompt_id} ({risk_domain})...")
    
    try:
        # Send the dangerous question to your Novita model
        completion = client.chat.completions.create(
            model=NOVITA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{adversarial_prompt}*"}
            ],
            temperature=0.0,
        )
        model_answer = completion.choices[0].message.content
        
    except Exception as e:
        model_answer = f"Error calling API: {e}"

    # Save EVERYTHING so we can do deep analysis later
    results.append({
        "ID": prompt_id,
        "risk_domain": risk_domain,
        "risk_subdomain": risk_subdomain,
        "adversarial_prompt": adversarial_prompt,
        "benign_prompt": benign_prompt,
        "your_system_prompt": SYSTEM_PROMPT,
        "model_answer": model_answer,
        "rubric": rubric 
    })

# ==========================================
# 4. SAVE THE ENRICHED RESULTS
# ==========================================
output_filename = "novita_test_results.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"\nDone! Enriched results saved to {output_filename}")