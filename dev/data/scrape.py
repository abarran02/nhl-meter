from pathlib import Path

import hockey_scraper
import pandas as pd

current_file_path = Path(__file__).resolve()
data_path = current_file_path.parent / '..' / '..' / 'data'

for y in range(2007,2024):
    try:
        scraped = hockey_scraper.scrape_seasons([y], True, data_format='pandas')
        # scraped = hockey_scraper.scrape_date_range('2017-12-08', '2017-12-08', True, data_format='pandas')  # includes four players with missing IDs for testing

        shifts = scraped['shifts']
        pbp = scraped['pbp']

        shifts['Period'] = pd.to_numeric(shifts['Period'], errors='coerce')
        pbp['Period'] = pd.to_numeric(pbp['Period'], errors='coerce')

        shifts.to_parquet(data_path / f'shift_{y}.parquet')
        pbp.to_parquet(data_path / f'game_{y}.parquet')

    except TimeoutError:
        continue
