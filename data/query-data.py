import sqlite3

#conn = sqlite3.connect("raw/raw_reddit_posts.db")  
conn = sqlite3.connect("reddit_data.db")  
cursor = conn.cursor()

# Example query: Get 5 posts from a specific subreddit
# subreddit_name = "news"
# cursor.execute('''
#     SELECT id, title, subreddit, score, timestamp
#     FROM raw_reddit_posts
#     WHERE subreddit = ?
#     LIMIT 5
# ''', (subreddit_name,))

# cursor.execute('''
#     SELECT *
#     FROM raw_reddit_posts
#     LIMIT 5
# ''')

# rows = cursor.fetchall()
# for row in rows:
#     print(row)



# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# tables = cursor.fetchall()

# # Print table names
# print("Tables in the database:")
# for table in tables:
#     print(f"• {table[0]}")


cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Show schema for each table
for table_name in tables:
    table_name = table_name[0]
    print(f"\n📘 Schema for table: {table_name}")
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for col in columns:
        cid, name, ctype, notnull, dflt_value, pk = col
        print(f" - {name} ({ctype}){' [PK]' if pk else ''}")


conn.close()
