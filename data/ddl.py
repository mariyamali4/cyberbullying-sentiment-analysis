import sqlite3

def create_tables(conn):
    cursor = conn.cursor()

    # Raw table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_reddit_posts (
            id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            score INTEGER,
            comments INTEGER,
            subreddit TEXT,
            timestamp TEXT
        );
    ''')

    # Cleaned table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cleaned_reddit_posts (
            id INTEGER PRIMARY KEY,
            cleaned_title TEXT,
            cleaned_at TIMESTAMP
        );
    ''')

    # Cleaned dimension table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cleaned_reddit_posts_dim (
            id INTEGER PRIMARY KEY,
            url TEXT,
            score INTEGER,
            comments INTEGER,
            subreddit TEXT
        );
    ''')

    # Labelled table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS labelled_reddit_posts (
            id INTEGER PRIMARY KEY,
            title TEXT,
            label INTEGER,
            labelled_at TIMESTAMP
        );
    ''')

    conn.commit()
    return conn


if __name__ == "__main__":
    conn = sqlite3.connect("../data/reddit_data.db")
    create_tables(conn)
    conn.close()
