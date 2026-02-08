import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from google import genai

# GEMINI SETUP 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load EMBEDDING MODEL 
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# LOAD DATABASES
print("Loading medical DB...")
medical_index = faiss.read_index("medical_index.faiss")
with open("medical_chunks.pkl", "rb") as f:
    medical_chunks = pickle.load(f)

print("Loading insurance DB...")
insurance_index = faiss.read_index("insurance_index.faiss")
with open("insurance_chunks.pkl", "rb") as f:
    insurance_chunks = pickle.load(f)

print("\nAI READY\n")


# VECTOR RETRIEVAL 
def retrieve(query: str, k: int = 4) -> str:
    q_emb = embed_model.encode([query])

    _, med_idx = medical_index.search(q_emb, k)
    _, ins_idx = insurance_index.search(q_emb, k)

    context_parts = []

    # --- Medical ---
    med_text = []
    for i in med_idx[0]:
        chunk = medical_chunks[i]
        med_text.append(chunk["text"] if isinstance(chunk, dict) else chunk)

    context_parts.append("[MEDICAL KNOWLEDGE]\n" + "\n\n".join(med_text))

    # --- Insurance ---
    ins_text = []
    for i in ins_idx[0]:
        chunk = insurance_chunks[i]
        ins_text.append(chunk["text"] if isinstance(chunk, dict) else chunk)

    context_parts.append("[INSURANCE KNOWLEDGE]\n" + "\n\n".join(ins_text))

    return "\n\n".join(context_parts)


# GEMINI RESPONSE 
def generate_answer(question: str, context: str) -> str:

    prompt = f"""
You are a Digital Medical Representative AI for licensed Healthcare Professionals (HCPs) in India.

You must answer the QUESTION using ONLY the provided CONTEXT.
The CONTEXT is the sole source of truth.
Do NOT use outside knowledge, assumptions, or general medical knowledge.

You are a scientific, non-promotional pharma representative.
Do NOT give diagnosis, treatment advice, or patient-specific guidance.

----------------------------------------
RESPONSE LENGTH
----------------------------------------
Default: 2–3 sentences max  
If user explicitly asks for detail: up to 5 sentences  

----------------------------------------
CORE RULE
----------------------------------------
If answer exists explicitly in CONTEXT:
→ Extract and present only that information

If answer is missing or unclear:
→ Respond ONLY with:
"Information limited in available documents."

Do not infer, assume, extrapolate, or combine separate facts.

----------------------------------------
WHAT YOU MAY INCLUDE (only if in context)
----------------------------------------
• Mechanism of action  
• Indications  
• Dosage  
• Safety / side effects  
• Contraindications  
• Insurance / reimbursement info (India)  

----------------------------------------
INSURANCE RULES (CRITICAL)
----------------------------------------
Mention insurance ONLY if:
1) Explicitly present in context  
OR  
2) User directly asks about insurance

Never just assume:
• coverage  
• reimbursement  
• eligibility  
• formulary inclusion  

If type of insurance not specified by user, do a one liner-comparision between public and private policies
If insurance asked but not in context:
Either mention the general insurance policies that could apply to the context or if too ambiguous and assumptions are a reach, mention "Information limited in available documents."

----------------------------------------
TONE
----------------------------------------
Professional  
Neutral  
Scientific  
Concise  
Non-promotional  

----------------------------------------
OUTPUT FORMAT (use only relevant sections)
----------------------------------------

**Overview**  
(1–2 lines)

**Clinical / Scientific Details**  
(bullets only if present)

**Insurance & Access (India)**  
(only if present or asked)

**Key Takeaway for HCP**  
(1 short factual line)

----------------------------------------

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


# ================= CHAT LOOP =================
print("👋 Hello! I'm your Digital Medical Representative AI.")
print("Ask me about drugs, clinical data, or insurance coverage.\n")

while True:
    query = input("Ask: ")

    if query.lower() in ["exit", "quit"]:
        print("Goodbye 👋")
        break

    context = retrieve(query)
    ans = generate_answer(query, context)

    print("\nAnswer:\n")
    print(ans)
    print("\n" + "=" * 60 + "\n")
