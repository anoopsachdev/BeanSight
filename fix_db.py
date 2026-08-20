import asyncio
from sqlalchemy import text, create_engine
from app.config import get_settings

def run():
    url = get_settings().DATABASE_URL
    # switch from asyncpg to psycopg2 style and port 6543 to 5432
    sync_url = url.replace('+asyncpg', '').replace(':6543', ':5432')
    
    engine = create_engine(sync_url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS predictions;"))
    
    from app.database import Base
    Base.metadata.create_all(engine)
    print("Table recreated via sync engine!")

run()
