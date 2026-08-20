import asyncio
from app.database import get_history
async def run():
    print(await get_history())
asyncio.run(run())
