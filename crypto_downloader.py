#!/usr/bin/env python3
"""
Bitvavo Crypto Data Downloader

Downloads historical OHLCV data from Bitvavo API based on configuration.
Supports multiple coins, incremental updates, and intelligent bulk/sequential downloading.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import pandas as pd
from python_bitvavo_api.bitvavo import Bitvavo


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.txt") -> Dict:
    """
    Read and parse JSON config file with validation.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary with configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['coins', 'start_date', 'interval']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate coins is a list
    if not isinstance(config['coins'], list) or len(config['coins']) == 0:
        raise ValueError("'coins' must be a non-empty list")
    
    # Validate date format
    try:
        datetime.strptime(config['start_date'], '%Y-%m-%d')
        if 'end_date' in config and config['end_date']:
            datetime.strptime(config['end_date'], '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
    
    # Set default end_date to current date if not specified
    if 'end_date' not in config or not config['end_date']:
        config['end_date'] = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"end_date not specified, using current date: {config['end_date']}")
    
    # Set default logging level
    if 'logging_level' not in config:
        config['logging_level'] = 'INFO'
    
    # Set logging level
    log_level = getattr(logging, config['logging_level'].upper(), logging.INFO)
    logger.setLevel(log_level)
    logging.getLogger().setLevel(log_level)
    
    return config


def get_csv_filename(coin: str, interval: str, start_date: str, end_date: str) -> str:
    """
    Generate CSV filename in format: COIN_INTERVAL_STARTDATE-ENDDATE.csv
    
    Args:
        coin: Coin symbol (e.g., "BTC-EUR")
        interval: Interval (e.g., "1d")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Filename string
    """
    # Convert interval to uppercase
    interval_upper = interval.upper()
    
    # Convert dates to YYYYMMDD format
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_str = start_dt.strftime('%Y%m%d')
    end_str = end_dt.strftime('%Y%m%d')
    
    return f"{coin}_{interval_upper}_{start_str}-{end_str}.csv"


def load_existing_data(csv_path: str) -> pd.DataFrame:
    """
    Load existing CSV data and return as DataFrame.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with existing data, or empty DataFrame if file doesn't exist
    """
    if not os.path.exists(csv_path):
        logger.info(f"CSV file does not exist: {csv_path}")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    try:
        df = pd.read_csv(csv_path)
        if 'timestamp' not in df.columns:
            logger.warning(f"CSV file missing 'timestamp' column: {csv_path}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Ensure timestamp is numeric
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        logger.info(f"Loaded {len(df)} existing records from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


def find_missing_periods(
    existing_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    interval: str
) -> List[Tuple[int, int]]:
    """
    Identify missing time periods in the data.
    
    Args:
        existing_df: DataFrame with existing data
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)
        interval: Candle interval (e.g., "1d", "1h")
        
    Returns:
        List of tuples (start_timestamp_ms, end_timestamp_ms) for missing periods
    """
    if existing_df.empty:
        # No existing data, need entire range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        # Set end to last possible time of the day (23:59:59.999)
        end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999000)
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        return [(start_ts, end_ts)]
    
    # Get existing timestamps
    existing_timestamps = set(existing_df['timestamp'].astype(int).tolist())
    
    # Calculate interval in milliseconds
    interval_ms = _interval_to_milliseconds(interval)
    
    # Generate expected timestamps
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999000)
    
    current_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    missing_periods = []
    gap_start = None
    
    while current_ts <= end_ts:
        if current_ts not in existing_timestamps:
            if gap_start is None:
                gap_start = current_ts
        else:
            if gap_start is not None:
                # End of gap
                missing_periods.append((gap_start, current_ts - interval_ms))
                gap_start = None
        current_ts += interval_ms
    
    # Handle gap at the end
    if gap_start is not None:
        missing_periods.append((gap_start, end_ts))
    
    if not missing_periods:
        logger.info("No missing periods found")
    else:
        logger.info(f"Found {len(missing_periods)} missing period(s)")
    
    return missing_periods


def _interval_to_milliseconds(interval: str) -> int:
    """Convert interval string to milliseconds."""
    multipliers = {
        'm': 60 * 1000,  # minutes
        'h': 60 * 60 * 1000,  # hours
        'd': 24 * 60 * 60 * 1000,  # days
        'w': 7 * 24 * 60 * 60 * 1000  # weeks
    }
    
    if interval[-1] not in multipliers:
        raise ValueError(f"Invalid interval: {interval}")
    
    number = int(interval[:-1])
    unit = interval[-1]
    return number * multipliers[unit]


def retry_request(func, max_retries: int = 3, *args, **kwargs):
    """
    Wrapper function with retry logic and exponential backoff.
    
    Args:
        func: Function to call
        max_retries: Maximum number of retries
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Result from func
        
    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            # Check if it's a rate limit error
            error_str = str(e).lower()
            if '429' in error_str or 'rate limit' in error_str:
                # Wait longer for rate limit errors
                wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                logger.warning(f"Rate limit error, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
            else:
                wait_time = (2 ** attempt) * 1  # 1s, 2s, 4s
                logger.warning(f"Error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {wait_time}s...")
            
            time.sleep(wait_time)
    
    raise Exception(f"Failed after {max_retries} retries")


def download_data_bulk(
    bitvavo_client,
    coin: str,
    interval: str,
    start_ts: int,
    end_ts: int
) -> List[List]:
    """
    Attempt to download data in bulk for a period.
    Automatically chunks large date ranges to respect API limits (~1000 candles per request).
    
    Args:
        bitvavo_client: Bitvavo API client
        coin: Coin symbol
        interval: Candle interval
        start_ts: Start timestamp in milliseconds
        end_ts: End timestamp in milliseconds
        
    Returns:
        List of candle data: [[timestamp, open, high, low, close, volume], ...]
    """
    interval_ms = _interval_to_milliseconds(interval)
    
    # Calculate expected number of candles
    expected_candles = (end_ts - start_ts) // interval_ms + 1
    
    # Bitvavo API limit is typically around 1000-1500 candles per request
    # Use 1000 as a safe limit to avoid hitting the limit
    MAX_CANDLES_PER_REQUEST = 1000
    
    # If the range is small enough, try a single request
    if expected_candles <= MAX_CANDLES_PER_REQUEST:
        def _fetch():
            options = {
                'start': start_ts,
                'end': end_ts
            }
            response = bitvavo_client.candles(coin, interval, options)
            # Check if response is an error dict
            if isinstance(response, dict) and 'error' in response:
                raise Exception(f"API error: {response.get('error', 'Unknown error')}")
            return response
        
        try:
            data = retry_request(_fetch)
            # Ensure data is a list
            if not isinstance(data, list):
                logger.warning(f"Unexpected response type for {coin}: {type(data)}")
                return []
            if not data:
                logger.warning(f"No data returned for {coin}")
                return []
            
            # Check if we got all expected data
            if len(data) < expected_candles * 0.9:  # Allow 10% tolerance
                logger.warning(f"Received {len(data)} candles but expected ~{expected_candles}. "
                             f"API may have limited the response. Will chunk the request.")
                # Fall through to chunked download
            else:
                logger.info(f"Downloaded {len(data)} candles for {coin} (bulk)")
                return data
        except Exception as e:
            logger.warning(f"Bulk download failed for {coin}: {e}")
            return []
    
    # Chunk the request into smaller pieces
    # Work forwards chronologically from start_ts
    logger.info(f"Large date range detected (~{expected_candles} candles). "
                f"Chunking into requests of {MAX_CANDLES_PER_REQUEST} candles each.")
    
    all_data = []
    current_start_ts = start_ts
    chunk_num = 1
    seen_timestamps = set()
    max_chunks = (expected_candles // MAX_CANDLES_PER_REQUEST) + 10  # Safety limit
    
    while current_start_ts <= end_ts and chunk_num <= max_chunks:
        # Calculate chunk end timestamp
        chunk_end_ts = min(current_start_ts + (MAX_CANDLES_PER_REQUEST - 1) * interval_ms, end_ts)
        
        def _fetch_chunk():
            options = {
                'start': current_start_ts,
                'end': chunk_end_ts
            }
            response = bitvavo_client.candles(coin, interval, options)
            # Check if response is an error dict
            if isinstance(response, dict) and 'error' in response:
                raise Exception(f"API error: {response.get('error', 'Unknown error')}")
            return response
        
        try:
            chunk_data = retry_request(_fetch_chunk)
            if isinstance(chunk_data, list) and chunk_data:
                # Filter out duplicates (in case of overlap)
                new_candles = []
                for candle in chunk_data:
                    ts = int(candle[0])
                    if ts not in seen_timestamps and start_ts <= ts <= end_ts:
                        seen_timestamps.add(ts)
                        new_candles.append(candle)
                
                if new_candles:
                    all_data.extend(new_candles)
                    logger.info(f"Downloaded chunk {chunk_num}: {len(new_candles)} new candles "
                              f"(total: {len(all_data)})")
                    
                    # Move to next chunk: start from the last timestamp + 1 interval
                    last_ts = max(int(c[0]) for c in new_candles)
                    current_start_ts = last_ts + interval_ms
                else:
                    logger.info(f"Chunk {chunk_num}: all candles were duplicates or out of range, skipping")
                    # Move forward by chunk size if no new data
                    current_start_ts = chunk_end_ts + interval_ms
            else:
                logger.warning(f"Chunk {chunk_num} returned no data")
                # Move forward by chunk size if no data
                current_start_ts = chunk_end_ts + interval_ms
            
            # Small delay between chunks to avoid rate limits
            time.sleep(0.2)
            chunk_num += 1
            
        except Exception as e:
            logger.warning(f"Chunk {chunk_num} download failed for {coin}: {e}")
            # Move forward even on error to avoid infinite loop
            current_start_ts = chunk_end_ts + interval_ms
            chunk_num += 1
            continue
    
    if all_data:
        logger.info(f"Downloaded {len(all_data)} total candles for {coin} (chunked bulk, {chunk_num-1} chunks)")
    return all_data


def download_data_sequential(
    bitvavo_client,
    coin: str,
    interval: str,
    start_ts: int,
    end_ts: int
) -> List[List]:
    """
    Download data one interval at a time (fallback method).
    
    Args:
        bitvavo_client: Bitvavo API client
        coin: Coin symbol
        interval: Candle interval
        start_ts: Start timestamp in milliseconds
        end_ts: End timestamp in milliseconds
        
    Returns:
        List of candle data: [[timestamp, open, high, low, close, volume], ...]
    """
    interval_ms = _interval_to_milliseconds(interval)
    all_data = []
    current_ts = start_ts
    
    logger.info(f"Starting sequential download for {coin} from {start_ts} to {end_ts}")
    
    while current_ts <= end_ts:
        candle_end = min(current_ts + interval_ms - 1, end_ts)
        
        def _fetch_single():
            options = {
                'start': current_ts,
                'end': candle_end
            }
            response = bitvavo_client.candles(coin, interval, options)
            # Check if response is an error dict
            if isinstance(response, dict) and 'error' in response:
                raise Exception(f"API error: {response.get('error', 'Unknown error')}")
            return response
        
        try:
            data = retry_request(_fetch_single)
            # Ensure data is a list
            if isinstance(data, list) and data:
                all_data.extend(data)
            elif isinstance(data, dict) and 'error' in data:
                logger.error(f"API error for {coin} at {current_ts}: {data.get('error')}")
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
            current_ts += interval_ms
        except Exception as e:
            logger.error(f"Failed to download candle at {current_ts}: {e}")
            current_ts += interval_ms
            continue
    
    logger.info(f"Downloaded {len(all_data)} candles for {coin} (sequential)")
    return all_data


def validate_data(data: List[List], coin: str, interval: str, expected_start: int, expected_end: int) -> Tuple[List[List], List[Tuple[int, int]]]:
    """
    Validate timestamps and log missing data.
    
    Args:
        data: List of candle data
        coin: Coin symbol
        interval: Candle interval (e.g., "1d", "1h")
        expected_start: Expected start timestamp
        expected_end: Expected end timestamp
        
    Returns:
        Tuple of (validated_data, missing_periods)
    """
    if not data:
        logger.warning(f"No data received for {coin}")
        return [], [(expected_start, expected_end)]
    
    validated = []
    timestamps = []
    missing_periods = []
    
    for candle in data:
        if len(candle) < 6:
            logger.warning(f"Invalid candle data: {candle}")
            continue
        
        ts = int(candle[0])
        timestamps.append(ts)
        validated.append(candle)
    
    if not validated:
        return [], [(expected_start, expected_end)]
    
    # Sort by timestamp
    validated.sort(key=lambda x: x[0])
    timestamps.sort()
    
    # Calculate interval in milliseconds
    interval_ms = _interval_to_milliseconds(interval)
    
    # Check for gaps within the data
    for i in range(len(timestamps) - 1):
        gap = timestamps[i + 1] - timestamps[i]
        if gap > interval_ms * 1.5:  # Allow some tolerance
            gap_start = timestamps[i] + interval_ms
            gap_end = timestamps[i + 1] - interval_ms
            missing_periods.append((gap_start, gap_end))
            logger.warning(f"Missing data for {coin} between {gap_start} and {gap_end}")
    
    # Check if we have data at the boundaries
    if timestamps[0] > expected_start:
        missing_start = expected_start
        missing_end = timestamps[0] - interval_ms
        if missing_end >= missing_start:
            missing_periods.append((missing_start, missing_end))
            logger.warning(f"Missing data for {coin} at start: {missing_start} to {missing_end}")
    
    if timestamps[-1] < expected_end:
        missing_start = timestamps[-1] + interval_ms
        missing_end = expected_end
        if missing_end >= missing_start:
            missing_periods.append((missing_start, missing_end))
            logger.warning(f"Missing data for {coin} at end: {missing_start} to {missing_end}")
    
    return validated, missing_periods


def merge_and_save(
    existing_df: pd.DataFrame,
    new_data: List[List],
    csv_path: str,
    coin: str
) -> int:
    """
    Merge existing and new data, remove duplicates (keep last), sort, and save to CSV.
    
    Args:
        existing_df: Existing DataFrame
        new_data: New candle data
        csv_path: Path to CSV file
        coin: Coin symbol
        
    Returns:
        Number of new records added
    """
    if not new_data:
        logger.info(f"No new data to merge for {coin}")
        if not existing_df.empty:
            # Save existing data even if no new data
            existing_df.to_csv(csv_path, index=False)
        return 0
    
    # Convert new data to DataFrame
    new_df = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # Convert all columns to numeric (API returns strings)
    for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    new_df = new_df.dropna(subset=['timestamp'])
    
    # Combine existing and new data
    if existing_df.empty:
        combined_df = new_df
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates based on timestamp, keep last occurrence
    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    combined_df.to_csv(csv_path, index=False)
    
    new_count = len(new_df)
    total_count = len(combined_df)
    logger.info(f"Saved {total_count} total records ({new_count} new) to {csv_path}")
    
    return new_count


def generate_summary(results: Dict) -> str:
    """
    Generate short summary of download results.
    
    Args:
        results: Dictionary with results per coin
        
    Returns:
        Summary string
    """
    summary_lines = [
        "\n" + "="*60,
        "DOWNLOAD SUMMARY",
        "="*60
    ]
    
    total_coins = len(results)
    total_records = sum(r.get('records_downloaded', 0) for r in results.values())
    total_missing = sum(len(r.get('missing_periods', [])) for r in results.values())
    total_errors = sum(r.get('errors', 0) for r in results.values())
    
    summary_lines.append(f"Total coins processed: {total_coins}")
    summary_lines.append(f"Total records downloaded: {total_records}")
    summary_lines.append(f"Missing data periods: {total_missing}")
    summary_lines.append(f"Errors encountered: {total_errors}")
    summary_lines.append("")
    summary_lines.append("Per coin details:")
    
    for coin, result in results.items():
        summary_lines.append(f"  {coin}:")
        summary_lines.append(f"    - Records: {result.get('records_downloaded', 0)}")
        summary_lines.append(f"    - Missing periods: {len(result.get('missing_periods', []))}")
        summary_lines.append(f"    - File: {result.get('file', 'N/A')}")
        if result.get('errors', 0) > 0:
            summary_lines.append(f"    - Errors: {result.get('errors', 0)}")
    
    summary_lines.append("="*60 + "\n")
    
    return "\n".join(summary_lines)


def main():
    """Main orchestration function."""
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Loaded configuration: {len(config['coins'])} coin(s), interval: {config['interval']}")
        
        # Create datasets folder if it doesn't exist
        datasets_folder = "datasets"
        os.makedirs(datasets_folder, exist_ok=True)
        logger.info(f"Using datasets folder: {datasets_folder}")
        
        # Initialize Bitvavo client (no API keys needed for public endpoint)
        bitvavo_client = Bitvavo({})
        
        # Process each coin
        results = {}
        
        for coin in config['coins']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {coin}")
            logger.info(f"{'='*60}")
            
            coin_results = {
                'records_downloaded': 0,
                'missing_periods': [],
                'errors': 0,
                'file': None
            }
            
            try:
                # Generate CSV filename
                csv_filename = get_csv_filename(
                    coin,
                    config['interval'],
                    config['start_date'],
                    config['end_date']
                )
                # Save to datasets folder
                csv_path = os.path.join(datasets_folder, csv_filename)
                coin_results['file'] = csv_path
                
                # Load existing data
                existing_df = load_existing_data(csv_path)
                
                # Find missing periods
                missing_periods = find_missing_periods(
                    existing_df,
                    config['start_date'],
                    config['end_date'],
                    config['interval']
                )
                
                if not missing_periods:
                    logger.info(f"No missing data for {coin}, skipping download")
                    results[coin] = coin_results
                    continue
                
                # Download missing data
                all_new_data = []
                all_missing = []
                
                for start_ts, end_ts in missing_periods:
                    logger.info(f"Downloading data for {coin} from {start_ts} to {end_ts}")
                    
                    # Try bulk download first
                    data = download_data_bulk(
                        bitvavo_client,
                        coin,
                        config['interval'],
                        start_ts,
                        end_ts
                    )
                    
                    # If bulk failed or returned no data, try sequential
                    if not data:
                        logger.info(f"Bulk download failed or empty, trying sequential for {coin}")
                        data = download_data_sequential(
                            bitvavo_client,
                            coin,
                            config['interval'],
                            start_ts,
                            end_ts
                        )
                    
                    # Validate data
                    validated_data, missing = validate_data(data, coin, config['interval'], start_ts, end_ts)
                    all_new_data.extend(validated_data)
                    all_missing.extend(missing)
                
                # Merge and save
                new_count = merge_and_save(existing_df, all_new_data, csv_path, coin)
                coin_results['records_downloaded'] = new_count
                coin_results['missing_periods'] = all_missing
                
            except Exception as e:
                logger.error(f"Error processing {coin}: {e}", exc_info=True)
                coin_results['errors'] = 1
            
            results[coin] = coin_results
        
        # Generate and print summary
        summary = generate_summary(results)
        print(summary)
        logger.info("Download process completed")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

