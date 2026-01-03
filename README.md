# Bitvavo Crypto Data Downloader

A Python script that downloads historical cryptocurrency OHLCV (Open, High, Low, Close, Volume) data from Bitvavo API.

## Features

- **Multiple Coins Support**: Download data for multiple coins with the same interval in one run
- **Incremental Updates**: Only downloads missing data, merges with existing CSV files
- **Intelligent Downloading**: Attempts bulk downloads first, falls back to sequential if needed
- **Error Handling**: 3-retry limit with exponential backoff, rate limit management
- **Data Validation**: Validates timestamps and logs missing data periods
- **Duplicate Handling**: Automatically removes timestamp-based duplicates (keeps last)

## Installation

1. Install Python 3.7 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.txt` to specify your download parameters:

```json
{
  "coins": ["BTC-EUR", "ETH-EUR"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "interval": "1d",
  "logging_level": "INFO"
}
```

### Configuration Fields

- `coins`: Array of coin symbols (e.g., `["BTC-EUR", "ETH-EUR"]`)
- `start_date`: Start date in `YYYY-MM-DD` format (inclusive)
- `end_date`: End date in `YYYY-MM-DD` format (inclusive). If not specified, uses current date
- `interval`: Candle interval - supported: `1m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `1w`
- `logging_level`: Optional - `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

**Note**: No API keys required - `getCandles` is a public endpoint.

## Usage

```bash
python crypto_downloader.py
```

## Output

CSV files are named in the format: `{COIN}_{INTERVAL}_{STARTDATE}-{ENDDATE}.csv`

Example: `BTC-EUR_1D_20240101-20241231.csv`

### CSV Format

- Headers: `timestamp,open,high,low,close,volume`
- Timestamp: UTC timestamps in milliseconds (Unix epoch)
- All values are numeric

## Features Details

### Incremental Updates
The script checks existing CSV files and only downloads missing data periods. This allows you to:
- Run the script multiple times to update data
- Resume interrupted downloads
- Add new date ranges without re-downloading existing data

### Rate Limit Management
- Monitors Bitvavo API rate limits (1000 weight points per minute)
- Automatically adds delays when approaching limits
- Implements exponential backoff on rate limit errors

### Error Handling
- 3 retry attempts per request with exponential backoff
- Handles network errors, API errors, and rate limits
- Logs all errors for debugging

### Missing Data
Missing data periods are logged as warnings. The script will:
- Skip missing periods
- Log warnings with timestamps
- Include missing data count in summary

## Summary Output

At the end of execution, a summary is displayed showing:
- Total coins processed
- Records downloaded per coin
- Missing data periods
- Errors encountered
- Files created/updated




