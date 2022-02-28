import sqlite3
from pathlib import Path
import yaml
from imutils.paths import list_images
import sys

def dirty_train_val(data_path,ratio,source):
    imgs=[]
    for img_dir in source:
        imgs+=list(list_images((Path(data_path)/img_dir)))
    n=len(imgs)
    m=int(ratio*n)
    train_part=imgs[:m]
    val_part=imgs[m:]
    return train_part,val_part
    
def write_into_sqt(stamp):
    with open('data/data_record.yaml','r') as f:
        record=yaml.safe_load(f)
    source=record[stamp]
    train_part,val_part=dirty_train_val('data/',0.9,source)
    conn = sqlite3.connect('data/train_source.db')
    c = conn.cursor()
    c.execute(f'''CREATE TABLE IF NOT EXISTS  '{stamp}'
       (ID  PRIMARY KEY     NOT NULL,
       PATH           TEXT    NOT NULL,
       STAGE          TEXT    NOT NULL);''')
    for x in train_part:
        c.execute(f"INSERT INTO '{stamp}' (ID,PATH,STAGE) \
            VALUES ('{str(Path(x).stem)}', '{str(x)}', 'train')")
    for x in val_part:
        c.execute(f"INSERT INTO '{stamp}' (ID,PATH,STAGE) \
            VALUES ('{str(Path(x).stem)}', '{str(x)}', 'val')")
    conn.commit()
    conn.close()
    print(f"{stamp} is written into sql")
    
def check():
    conn = sqlite3.connect('data/train_source.db')
    c = conn.cursor()
    stamp=''
    cursor =c.execute(f"SELECT * FROM '{stamp}' WHERE STAGE='train'")
    for i,res in enumerate(cursor):
        print(res)
        if i>6:break
    
if __name__=='__main__':
    stamp=sys.argv[1]
    write_into_sqt(stamp)
    #check()
    
    