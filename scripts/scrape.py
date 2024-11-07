import hockey_scraper
import pandas as pd

if __name__ == "__main__":
    for y in range(2007,2024):
        try:
            scraped = hockey_scraper.scrape_seasons([y], True, data_format="pandas")
            # scraped = hockey_scraper.scrape_date_range('2017-12-08', '2017-12-08', True, data_format="pandas")  # includes four players with missing IDs for testing

            shifts = scraped["shifts"]
            pbp = scraped["pbp"]

            shifts['Period'] = pd.to_numeric(shifts['Period'], errors='coerce')
            pbp['Period'] = pd.to_numeric(pbp['Period'], errors='coerce')

            shifts.to_parquet(f"shift_{y}.parquet")
            pbp.to_parquet(f"game_{y}.parquet")
        except TimeoutError:
            continue
