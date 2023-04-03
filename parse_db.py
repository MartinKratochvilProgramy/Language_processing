import sqlite3
import re
from utils.train_split import train_split

con = sqlite3.connect("database.sqlite")
cur = con.cursor()

LIMIT = 200
SCORE = 5
sentiment = "positive" if (SCORE == 4 or SCORE == 5) else "negative"

res = cur.execute(f"SELECT Text FROM Reviews WHERE Score={SCORE} LIMIT {LIMIT};")
data = res.fetchall()

for i, review in enumerate(data):
    route = f"./datasets/train/class_{sentiment}/reviews_{sentiment}_{i+1}.txt"
    f = open(route, "w")
    
    text = str(review[0])
    formatted_text = re.sub('[^0-9a-zA-Z. -]', '', text)

    f.write(formatted_text)
    f.close()

    print(f"Wrote {route}")

if sentiment == "positive":
    train_split('./datasets/train/class_positive', './datasets/test/class_positive', 0.2)
elif sentiment == "negative":    
    train_split('./datasets/train/class_negative', './datasets/test/class_negative', 0.2)