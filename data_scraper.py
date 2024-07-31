from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
from io import StringIO
from datetime import datetime
import random
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballDataScraper:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.create_directories()
        self.chrome_options = self.set_chrome_options()

    def create_directories(self):
        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def set_chrome_options():
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-gpu")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")
        return options

    def get_driver(self):
        return webdriver.Chrome(options=self.chrome_options)

    def scrape_tables(self, driver, url, table_ids):
        logger.info(f"Scraping tables from {url}")
        try:
            driver.get(url)

            # Wait for elements to load
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Close ad if present
            try:
                driver.execute_script("""
                    var closeButton = document.querySelector('[class*="fs-close-button fs-close-button-sticky"]');
                    if (closeButton) closeButton.click();
                """)
            except Exception as e:
                logger.info(f"No ad to close: {str(e)}")

            dfs = []
            for table_id in table_ids:
                # Toggle to per match view if available
                try:
                    toggle_script = f"""
                        var toggle = document.getElementById('{table_id}_per_match_toggle');
                        if (toggle) {{
                            toggle.scrollIntoView();
                            toggle.click();
                        }}
                    """
                    driver.execute_script(toggle_script)
                except Exception as e:
                    logger.info(f"No per match toggle available for {table_id}: {str(e)}")

                try:
                    table_html = WebDriverWait(driver, 60).until(
                        EC.presence_of_element_located((By.ID, table_id))
                    ).get_attribute('outerHTML')

                    df = pd.read_html(StringIO(table_html))[0]

                    # Flatten multi-level columns to single level if unique, otherwise combine them
                    if df.columns.nlevels > 1:
                        df.columns = [self.flatten_column(col, df.columns) for col in df.columns.values]

                    # Rename 'Squad' column if necessary
                    df.rename(columns={col: 'Squad' for col in df.columns if 'Squad' in col}, inplace=True)

                    dfs.append(df)
                except TimeoutException:
                    logger.warning(f"Table {table_id} not found on page.")

                if not dfs:
                    raise ValueError("No tables were found for this season.")

            merged_df = dfs[0]
            for df in dfs[1:]:
                common_cols = set(merged_df.columns).intersection(set(df.columns))
                for col in common_cols:
                    if col != 'Squad' and not merged_df[col].equals(df[col]):
                        df.rename(columns={col: f"{col}_{df.columns.name}"}, inplace=True)
                merged_df = merged_df.merge(df, on="Squad", suffixes=('', '_drop'))
                merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_drop')]

            return merged_df

        except (TimeoutException, NoSuchElementException) as e:
            logger.error(f"Error scraping tables from {url}: {str(e)}")
            raise

    @staticmethod
    def flatten_column(col, columns):
        if isinstance(col, tuple):  # if multi-level columns
            col = [c for c in col if c]
            # Check if last element is unique
            col_name = col[-1]
            other_columns = [c[-1] for c in columns if c[-1] == col_name]
            if len(other_columns) > 1:  # if not unique, combine all elements
                return '_'.join(col)
            return col_name  # if unique, return last element
        return col

    @staticmethod
    def get_table_ids(season):
        table_ids = [
            "stats_squads_standard_for",
            "stats_squads_keeper_for",
            "stats_squads_shooting_for",
            "stats_squads_playing_time_for",
            "stats_squads_misc_for",
            "stats_squads_keeper_adv_for",
            "stats_squads_passing_types_for",
            "stats_squads_passing_for",
            "stats_squads_gca_for",
            "stats_squads_defense_for",
            "stats_squads_possession_for",
        ]
        # Older seasons are missing some tables
        if season[:4] <= "2016":
            unavailable_tables = [
                "stats_squads_keeper_adv_for",
                "stats_squads_passing_types_for",
                "stats_squads_passing_for",
                "stats_squads_gca_for",
                "stats_squads_defense_for",
                "stats_squads_possession_for",
            ]
            table_ids = [table_id for table_id in table_ids if table_id not in unavailable_tables]
        return table_ids

    def scrape_season_data(self, season):
        url = f"https://fbref.com/en/comps/9/{season}/{season}-Premier-League-Stats"
        table_ids = self.get_table_ids(season)
        retries = 3

        for attempt in range(retries):
            try:
                with self.get_driver() as driver:
                    season_data = self.scrape_tables(driver, url, table_ids)
                    season_data['Season'] = season
                    return season_data
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for season {season}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(random.uniform(5, 10))  # Exponential backoff
                else:
                    return pd.DataFrame()

    def scrape_multiple_seasons(self, seasons):
        with ThreadPoolExecutor(max_workers=len(seasons)) as executor:
            all_data = list(executor.map(self.scrape_season_data, seasons))

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = self.clean_dataframe(combined_df)
        combined_df.to_csv(os.path.join(self.data_dir, f'football_data.csv'), index=False)
        return combined_df

    @staticmethod
    def clean_dataframe(df):
        for col in df.columns:
            if '_drop' in col:
                original_col = col.replace('_drop', '')
                if df[original_col].equals(df[col]):
                    df.drop(columns=[col], inplace=True)
                else:
                    df.rename(columns={col: f"{original_col}_dup"}, inplace=True)
        return df

    def scrape_current_season(self):
        current_year = datetime.now().year
        current_month = datetime.now().month
        current_season = f"{current_year - 1}-{current_year}"
        if current_month > 7:
            current_season = f"{current_year}-{current_year + 1}"
        current_season_data = self.scrape_season_data(current_season)
        current_season_data.to_csv(os.path.join(self.data_dir, f'current_season_data.csv'), index=False)
        return current_season_data


if __name__ == "__main__":
    scraper = FootballDataScraper()
    seasons = ["2022-2023", "2021-2022", "2020-2021", "2019-2020", "2018-2019", "2017-2018"]
    past_data = scraper.scrape_multiple_seasons(seasons)
    current_data = scraper.scrape_current_season()
    print(f"Scraped data for current season. Shape: {current_data.shape}")
    print(f"Scraped data for {len(seasons)} seasons. Shape: {past_data.shape}")
