import sqlite3

# Connect to the database
conn = sqlite3.connect("raw/raw_reddit_posts.db")  
cursor = conn.cursor()

# Example query: Get 5 posts from a specific subreddit
# subreddit_name = "news"
# cursor.execute('''
#     SELECT id, title, subreddit, score, timestamp
#     FROM raw_reddit_posts
#     WHERE subreddit = ?
#     LIMIT 5
# ''', (subreddit_name,))

cursor.execute('''
    SELECT count(*)
    FROM raw_reddit_posts
    -- LIMIT 5
''')

rows = cursor.fetchall()
for row in rows:
    print(row)

# Close connection
conn.close()
