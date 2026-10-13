from apscheduler.schedulers.blocking import BlockingScheduler
from app.pipeline.daily_pipeline import daily_pipeline
from app.pipeline.daily_pipeline import run_ingestion

if __name__ == "__main__":
    sched = BlockingScheduler()
    sched.add_job(run_ingestion, "interval", hours=2)
    sched.add_job(daily_pipeline, "cron", hour=6, minute=0)
    print("Scheduler running. Ctrl+C to stop.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("Stopped.")

# This sets up a scheduler to run two tasks at specified intervals. The `run_ingestion` function is scheduled to run every 2 hours, while the `daily_pipeline` function is scheduled to run daily at 6:00 AM. The script uses the `BlockingScheduler` from the `apscheduler` library to manage the scheduling of these tasks. When executed, it will continuously run the scheduler until interrupted by the user.