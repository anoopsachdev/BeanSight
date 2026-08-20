import asyncio
from app.database import init_db, log_prediction, get_history

async def run():
    await init_db()
    try:
        pid = await log_prediction(
            filename="test.jpg",
            analysis_type="roast",
            predicted_class="Dark",
            confidence=0.99,
            probabilities={"Dark": 0.99},
            inference_time_ms=10.0,
            image_url=None
        )
        print("Logged ID:", pid)
        history = await get_history()
        print("History:", history)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(run())
