from app.pipeline.daily_pipeline import run_ingestion

if __name__ == "__main__":
    run_ingestion()

# serves as the entry point for running the ingestion process of the "electronics-intel" project. It imports the `run_ingestion` function from the `daily_pipeline` module and executes it when the script is run directly. The ingestion process typically involves fetching and processing new data, such as articles or feeds, to be added to the system's database or index.
