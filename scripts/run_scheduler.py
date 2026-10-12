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
