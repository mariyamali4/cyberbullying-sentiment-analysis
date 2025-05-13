import pandas as pd
import openai
from tqdm import tqdm
import time

# === CONFIG ===
GROQ_API_KEY = "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"  #  Replace with your key or use env vars
MODEL_NAME = "llama3-70b-8192"
INPUT_CSV = "reddit_cleaned_posts.csv"
OUTPUT_CSV = "labeled_reddit.csv"

# === PROMPT TEMPLATE ===
def build_prompt(text):
    return f"""
You are a content moderation AI. Your task is to detect cyberhate in social media comments.

Cyberhate includes:
- Direct or indirect harassment
- Personal insults or threats
- Hate speech (racial, gendered, religious slurs)
- Abusive, degrading, or harmful language
- Encouragement of self-harm or violence
- Any other form of cyberbullying

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

Comment: "Maybe just kill yourself"
CyberHate: 1

Comment: "You should consider a healthy diet for a healthy lifestyle"
CyberHate: 0

Comment: "Have you considered losing all that weight, fatty?"
CyberHate: 1

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


def main():
    # === SETUP CLIENT ===
    client = openai.OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    input_csv = 'raw_reddit_scraped_posts.csv'
    output_csv = 'labeled_reddit_posts.csv'

    # === LOAD DATA ===
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["Title"]).reset_index(drop=True)
    df['CyberHate'] = None

    # === CLASSIFY ===
    for i in tqdm(range(len(df))):
        if pd.isna(df.loc[i, "CyberHate"]):
            df.loc[i, "CyberHate"] = classify_text(df.loc[i, "Title"], client)
            time.sleep(0.3)

    # === SAVE RESULTS ===
    df.to_csv(output_csv, index=False)
    print(f"✅ Done! Labeled file saved as: {output_csv}")


if __name__ == "__main__":
    main()
