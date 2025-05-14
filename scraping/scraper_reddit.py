import praw
import sqlite3
from datetime import datetime

# # Initialize Reddit client
# reddit = praw.Reddit(
#     client_id="IG1IzZGAdmS3FRTPyMndvA",
#     client_secret="tR2hKKi9-b430nYk6hl6iB8QwvVecw",
#     user_agent="MyRedditScraper/0.1 by u/csgeekkkkk",
# )

# # Subreddits to search in
# subreddits = ["all", "news", "politics", "worldnews", "askreddit", "socialjustice", "TrueOffMyChest"]

# # How many posts to collect in total
# total_limit = 10000


# def init_db():
#     conn = sqlite3.connect("../data/raw/raw_reddit_posts.db")
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS raw_reddit_posts (
#             id TEXT PRIMARY KEY,
#             title TEXT,
#             url TEXT,
#             score INTEGER,
#             comments INTEGER,
#             subreddit TEXT,
#             timestamp TEXT
#         )
#     ''')
#     conn.commit()
#     return conn

# def store_in_db(conn, submission, timestamp):
#     cursor = conn.cursor()
#     try:
#         cursor.execute('''
#             INSERT INTO raw_reddit_posts (id, title, url, score, comments, subreddit, timestamp)
#             VALUES (?, ?, ?, ?, ?, ?, ?)
#         ''', (
#             submission.id,
#             submission.title,
#             submission.url,
#             submission.score,
#             submission.num_comments,
#             submission.subreddit.display_name,
#             timestamp
#         ))
#         conn.commit()
#     except sqlite3.IntegrityError:
#         # Duplicate post ID
#         pass

# def scrape_reddit():
#     seen_ids = set()
#     count = 0
#     conn = init_db()

#     csv_file = "../data_processing/raw_reddit_scraped_posts.csv"
#     with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
#         writer = csv.writer(file)
#         writer.writerow(["Title", "URL", "Score", "Comments", "Subreddit", "Timestamp (UTC)"])

#         for subreddit_name in subreddits:
#             print(f"\n--- Searching in r/{subreddit_name} ---\n")
#             subreddit = reddit.subreddit(subreddit_name)

#             for submission in subreddit.hot(limit=2000):
#                 if submission.id in seen_ids:
#                     continue
#                 seen_ids.add(submission.id)

#                 # Convert Unix timestamp to readable datetime string (UTC)
#                 post_time = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')

#                 writer.writerow([
#                     submission.title,
#                     submission.url,
#                     submission.score,
#                     submission.num_comments,
#                     submission.subreddit.display_name,
#                     post_time
#                 ])

#                 store_in_db(conn, submission, post_time)

#                 count += 1

#                 print(f"{count}: Saved — {submission.title[:60]}...")

#                 if count >= total_limit:
#                     print(f"\n✅ Reached target of {total_limit} posts.")
#                     conn.close()
#                     return
#     conn.close()


# if __name__ == "__main__":
#     scrape_reddit()


# Connect to existing database (don't recreate the table)
#DB_PATH = "../data/reddit_data.db"

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # location of scraper_reddit.py
DB_PATH = os.path.join(BASE_DIR, "..", "data", "reddit_data.db")


conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Initialize Reddit client
reddit = praw.Reddit(
    client_id="IG1IzZGAdmS3FRTPyMndvA",
    client_secret="tR2hKKi9-b430nYk6hl6iB8QwvVecw",
    user_agent="MyRedditScraper/0.1 by u/csgeekkkkk",
)

# Define subreddits and limits
subreddits = ["news", "worldnews", "politics", "all", "pakistan", 'AskReddit', 'worldnews', 'memes', 'funny']
#total_limit = 10000 
total_limit = 100 

def scrape_reddit():
    seen_ids = set()
    count = 0

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)

        for submission in subreddit.hot(limit=200):
            if submission.id in seen_ids:
                continue
            seen_ids.add(submission.id)

            post_time = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')

            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO raw_reddit_posts 
                    (id, title, url, score, comments, subreddit, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    submission.id,
                    submission.title,
                    submission.url,
                    submission.score,
                    submission.num_comments,
                    submission.subreddit.display_name,
                    post_time
                ))
                conn.commit()
                count += 1
                print(f"{count}: Inserted: {submission.title[:60]}")

                if count >= total_limit:
                    print("✅ Reached scraping limit.")
                    return

            except Exception as e:
                print(f"❌ Scraping Error: {e}")
                continue

if __name__ == "__main__":
    scrape_reddit()
    conn.close()