import pandas as pd
import re

def is_clean_title(title):
    """
    Check if the title contains only letters, digits, basic punctuation, and spaces.
    """
    return bool(re.fullmatch(r"[A-Za-z0-9 .,!?'\-()]+", title))

def clean_reddit_titles(input_file="all_reddit_scraped_posts.csv", output_file="reddit_cleaned_posts.csv"):
    """
    Reads input_file, filters rows with special characters in the 'Title',
    and writes the clean data to output_file.
    """
    try:
        df = pd.read_csv(input_file)
        original_count = len(df)

        # Filter only rows with clean titles
        df_cleaned = df[df['Title'].apply(is_clean_title)]
        cleaned_count = len(df_cleaned)

        # Save to a new CSV file
        df_cleaned.to_csv(output_file, index=False)

        print(f"✅ Cleaned data saved to '{output_file}'.")
        print(f"🔢 {cleaned_count} out of {original_count} posts kept.")

    except FileNotFoundError:
        print(f"❌ File '{input_file}' not found.")
    except Exception as e:
        print(f"⚠️ Error during cleaning: {e}")

def main():
    clean_reddit_titles()

if __name__ == "__main__":
    main()