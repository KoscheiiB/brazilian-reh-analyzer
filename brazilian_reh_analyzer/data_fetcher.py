"""
Data fetching and caching components for Brazilian Central Bank APIs.
"""

import pandas as pd
import numpy as np
import requests
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from .utils import rate_limit_decorator, split_date_range

# Configure logging
logger = logging.getLogger(__name__)

# Check for BCB library availability
try:
    from bcb import sgs, Expectativas
    BCB_AVAILABLE = True
except ImportError:
    logger.warning("python-bcb not installed. Install with: pip install python-bcb")
    BCB_AVAILABLE = False


class DataCache:
    """
    Handles caching of fetched data to avoid repeated API calls
    """

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_filename(
        self, data_type: str, start_date: str, end_date: str
    ) -> Path:
        """Generate cache filename based on data type and date range"""
        safe_start = start_date.replace("-", "")
        safe_end = end_date.replace("-", "")
        timestamp = datetime.now().strftime("%Y%m%d")
        return (
            self.cache_dir / f"{data_type}_{safe_start}_to_{safe_end}_{timestamp}.pkl"
        )

    def _find_existing_cache(
        self, data_type: str, start_date: str, end_date: str, max_age_hours: int = 24
    ) -> Optional[Path]:
        """Find existing cache file that covers the date range and is recent enough"""
        pattern = f"{data_type}_*_to_*.pkl"

        for cache_file in self.cache_dir.glob(pattern):
            # Check if file is recent enough
            file_age = datetime.now() - datetime.fromtimestamp(
                cache_file.stat().st_mtime
            )
            if file_age.total_seconds() > max_age_hours * 3600:
                continue

            # Parse filename to check date coverage
            try:
                parts = cache_file.stem.split("_")
                if len(parts) >= 6:
                    cached_start = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}"
                    cached_end = f"{parts[3][:4]}-{parts[3][4:6]}-{parts[3][6:8]}"

                    # Check if cached range covers requested range
                    if cached_start <= start_date and cached_end >= end_date:
                        return cache_file
            except (ValueError, IndexError):
                continue

        return None

    def load_data(
        self, data_type: str, start_date: str, end_date: str, max_age_hours: int = 24
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and recent"""
        cache_file = self._find_existing_cache(
            data_type, start_date, end_date, max_age_hours
        )

        if cache_file is None:
            logger.info(f"No valid cache found for {data_type}")
            return None

        try:
            logger.info(f"Loading {data_type} from cache: {cache_file.name}")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            # Filter to exact date range requested
            if isinstance(data, pd.DataFrame) and "Data" in data.columns:
                data = data[(data["Data"] >= start_date) & (data["Data"] <= end_date)]
            elif isinstance(data, pd.Series):
                data = data[(data.index >= start_date) & (data.index <= end_date)]

            logger.info(f"Loaded {len(data)} records from cache")
            return data

        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
            return None

    def save_data(
        self, data: pd.DataFrame, data_type: str, start_date: str, end_date: str
    ):
        """Save data to cache"""
        cache_file = self._get_cache_filename(data_type, start_date, end_date)

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved {data_type} to cache: {cache_file.name}")

            # Clean up old cache files for this data type
            self._cleanup_old_cache(data_type, keep_latest=3)

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _cleanup_old_cache(self, data_type: str, keep_latest: int = 3):
        """Remove old cache files, keeping only the most recent ones"""
        pattern = f"{data_type}_*.pkl"
        cache_files = list(self.cache_dir.glob(pattern))

        if len(cache_files) > keep_latest:
            # Sort by modification time, newest first
            cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for old_file in cache_files[keep_latest:]:
                try:
                    old_file.unlink()
                    logger.debug(f"Removed old cache file: {old_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove old cache {old_file}: {e}")


class RespectfulBCBClient:
    """
    Wrapper for BCB API calls with built-in rate limiting and error handling
    """

    def __init__(self):
        if not BCB_AVAILABLE:
            raise ImportError(
                "python-bcb library required. Install with: pip install python-bcb"
            )
            
        self.session = requests.Session()
        # Add browser-like headers to avoid blocks
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "cross-site",
            }
        )

    @rate_limit_decorator(min_delay=0.8, max_delay=2.5, requests_per_batch=8)
    def get_sgs_data(self, series_code, start_date, end_date):
        """Rate-limited SGS data fetching"""
        return sgs.get(series_code, start=start_date, end=end_date)

    @rate_limit_decorator(min_delay=1.5, max_delay=4.0, requests_per_batch=5)
    def get_expectations_data(self, endpoint_name, query_params):
        """Rate-limited Expectations data fetching"""
        em = Expectativas()
        ep = em.get_endpoint(endpoint_name)

        # Build query step by step
        query = ep.query()

        # Apply filters
        if "filters" in query_params:
            for filter_condition in query_params["filters"]:
                query = query.filter(filter_condition)

        # Apply select
        if "select" in query_params:
            query = query.select(*query_params["select"])

        # Apply orderby
        if "orderby" in query_params:
            query = query.orderby(query_params["orderby"])

        # Apply limit
        if "limit" in query_params:
            query = query.limit(query_params["limit"])

        return query.collect()

    def fetch_ipca_data(self, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch IPCA 12-month accumulated data from BCB SGS API
        
        Parameters:
        - start_date: Start date in YYYY-MM-DD format
        - end_date: End date in YYYY-MM-DD format
        
        Returns:
        - pd.Series: IPCA 12-month accumulated data
        """
        logger.info("Fetching IPCA 12-month accumulated data from BCB...")

        try:
            # Use rate-limited client
            ipca_raw = self.get_sgs_data(433, start_date, end_date)

            # Clean and format data
            ipca_clean = ipca_raw.dropna()
            ipca_clean.name = "IPCA_12m"

            # Convert to monthly frequency using 'ME' instead of deprecated 'M'
            ipca_monthly = ipca_clean.resample("ME").last()

            logger.info(f"Fetched {len(ipca_monthly)} IPCA observations")
            return ipca_monthly

        except Exception as e:
            logger.error(f"Error fetching IPCA data: {e}")
            raise

    def fetch_focus_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch Focus Bulletin IPCA expectations data from BCB Expectations API
        
        Parameters:
        - start_date: Start date
        - end_date: End date
        
        Returns:
        - pd.DataFrame: Focus Bulletin expectations data
        """
        logger.info("Fetching Focus Bulletin expectations data from BCB...")

        try:
            # First, let's check available endpoints
            em = Expectativas()
            logger.info("Available endpoints:")
            em.describe()

            # Use the correct endpoint for 12-month inflation expectations
            ep = em.get_endpoint("ExpectativasMercadoInflacao12Meses")

            logger.info("Endpoint properties:")
            em.describe("ExpectativasMercadoInflacao12Meses")

            # Query with proper filters using the rate-limited client
            logger.info("Fetching data in batches to respect API limits...")

            # Split date range into chunks to avoid overwhelming the API
            date_ranges = split_date_range(start_date, end_date, months=12)
            all_data = []

            for start_chunk, end_chunk in date_ranges:
                logger.info(
                    f"Fetching data for {start_chunk.date()} to {end_chunk.date()}"
                )

                try:
                    # Build query for this chunk
                    chunk_data = (
                        ep.query()
                        .filter(ep.Indicador == "IPCA")
                        .filter(ep.Data >= start_chunk.strftime("%Y-%m-%d"))
                        .filter(ep.Data <= end_chunk.strftime("%Y-%m-%d"))
                        .filter(ep.Suavizada == "N")  # Non-smoothed
                        .select(ep.Data, ep.Mediana, ep.Media, ep.numeroRespondentes)
                        .orderby(ep.Data.desc())
                        .collect()
                    )

                    if len(chunk_data) > 0:
                        all_data.append(chunk_data)
                        logger.info(f"  -> Got {len(chunk_data)} records")
                    else:
                        logger.warning(f"  -> No data for this period")

                    # Rate limiting between chunks
                    import time
                    import random
                    time.sleep(random.uniform(2.0, 4.0))

                except Exception as e:
                    logger.warning(
                        f"Failed to fetch chunk {start_chunk.date()}-{end_chunk.date()}: {e}"
                    )
                    import time
                    import random
                    time.sleep(random.uniform(5.0, 10.0))  # Longer delay on error
                    continue

            if not all_data:
                raise ValueError("No Focus data could be retrieved")

            # Combine all chunks
            focus_raw = pd.concat(all_data, ignore_index=True)

            # Convert dates and clean data
            focus_raw["Data"] = pd.to_datetime(focus_raw["Data"])

            # Filter for quality (minimum number of respondents)
            focus_filtered = focus_raw[focus_raw["numeroRespondentes"] >= 10]

            # Remove duplicates and take latest for each date
            focus_filtered = focus_filtered.drop_duplicates(
                subset=["Data"], keep="last"
            )

            # Sort by date
            focus_filtered = focus_filtered.sort_values("Data").set_index("Data")

            logger.info(
                f"Successfully fetched {len(focus_filtered)} Focus observations"
            )

            return focus_filtered

        except Exception as e:
            logger.error(f"Error fetching Focus data: {e}")
            raise