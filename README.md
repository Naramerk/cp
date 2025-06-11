# Reddit Data Processor

This script processes Reddit data stored in a SQLite database, converting it to Parquet format and performing sentiment analysis and entity extraction using PySpark.

## Features

- Converts SQLite data to Parquet format for faster processing
- Performs sentiment analysis on posts (-1 to 1 scale)
- Detects mentions of countries (including adjective forms like "Russian", "American", etc.)
- Detects mentions of political leaders and their variations
- Updates the original SQLite database with the new analysis results

## Prerequisites

- Python 3.8+
- Java 8+ (required for PySpark)
- PySpark and its dependencies
- SQLite database with Reddit posts

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the following data files in the correct location:
- `src/data/countries.json`: Contains country information
- `src/data/leaders.json`: Contains leader information

## Usage

Run the script with the following command:

```bash
python process_reddit_data.py --sqlite-path /path/to/your/database.db --parquet-output /path/to/output/directory
```

Arguments:
- `--sqlite-path`: Path to your SQLite database containing Reddit posts
- `--parquet-output`: Directory where the Parquet files will be saved

## Data Processing

The script performs the following operations:
1. Loads data from SQLite database
2. Converts it to a Spark DataFrame
3. Performs sentiment analysis using PySpark ML
4. Detects country and leader mentions
5. Saves the processed data in Parquet format
6. Updates the original SQLite database with new analysis results

## Output

The script will:
1. Create Parquet files in the specified output directory
2. Update the following columns in the SQLite database:
   - `sentiment_score`: Float between -1 and 1
   - `country`: Country code (e.g., "US", "RU")
   - `leader`: Full name of the detected leader 