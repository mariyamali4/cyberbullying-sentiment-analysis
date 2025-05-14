import sqlite3
import openai
from tqdm import tqdm
import time

# === CONFIG ===
GROQ_API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"
DB_PATH = "../data/reddit_data.db"
TABLE_NAME = "labelled_reddit_posts"

# === PROMPT TEMPLATE ===
def build_prompt(text):
    return f"""
You are a content moderation AI. Your task is to detect cyberhate in social media comments.

Cyberhate includes:
- Direct or indirect harassment
- Personal insults or threats
- Hate speech (racial, gendered, religious slurs)
- Abusive, degrading, or harmful language

Neutral or harmless comments should be marked as not cyberhate.

Your output must be one of:
- 1 → Cyberhate
- 0 → Not Cyberhate

Do not explain or justify. Reply with ONLY the number 0 or 1.

### Examples:

Comment: "You're disgusting. No one wants you here."
CyberHate: 1

Comment: "I think cats are better than dogs."
CyberHate: 0

Comment: "Just shut up already, idiot."
CyberHate: 1

Comment: "This article made my day!"
CyberHate: 0

Comment: "{text}"
CyberHate:"""


def classify_text(text, client):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(text)}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return int(result) if result in ["0", "1"] else None
    except Exception as e:
        print(f"[Error] {text[:60]}... => {e}")
        return None


def label_data():
    # === SETUP CLIENT ===
    client = openai.OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    # === CONNECT TO DB ===
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # === FETCH UNLABELED DATA ===
    #cursor.execute(f"SELECT id, title FROM {TABLE_NAME} WHERE label IS NULL LIMIT 500")
    cursor.execute(f"SELECT id, title FROM cleaned_reddit_posts LIMIT 50")
    rows = cursor.fetchall()

    print(f"🔍 Found {len(rows)} unlabeled posts")

    # === CLASSIFY AND UPDATE ===
    for post_id, title in tqdm(rows):
        label = classify_text(title, client)
        if label is not None:
            # cursor.execute(f"UPDATE {TABLE_NAME} SET label = ? WHERE id = ?", (label, post_id))
            labelled_at = time.strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("""
                    INSERT INTO labelled_reddit_posts 
                    (id, title, label, labelled_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post_id,
                    title,
                    label,
                    labelled_at
                ))
            
            conn.commit()
            time.sleep(0.3)

    conn.close()
    print("✅ Done! Labels updated in the database.")


if __name__ == "__main__":
    label_data()