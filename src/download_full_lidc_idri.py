#!/usr/bin/env python3
from tcia_utils import nbia
import os
import logging
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
download_dir = "lidc-idri-data"
os.makedirs(download_dir, exist_ok=True)


def download_with_retry(series_data, max_retries=3):
    """Download with retry logic for failed series"""
    attempt = 0
    while attempt < max_retries:
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")
            nbia.downloadSeries(
                series_data=series_data,
                input_type="df",
                path=download_dir,
                number=10  # Download 10 series at a time
            )
            logger.info("Batch completed successfully")
            return True
        except Exception as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} failed: {str(e)}")
            if attempt < max_retries:
                wait_time = 30 * attempt
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts")
                return False
    return False


logger.info("=" * 60)
logger.info("Starting FULL LIDC-IDRI Collection Download (Robust Version)")
logger.info(f"Download directory: {os.path.abspath(download_dir)}")
logger.info(f"Start time: {datetime.now()}")
logger.info("=" * 60)

try:
    # Get all series data
    logger.info("Fetching series information from TCIA...")
    series_data = nbia.getSeries(collection="LIDC-IDRI", format="df")
    total_series = len(series_data)
    logger.info(f"Total series found: {total_series}")
    logger.info(f"Estimated size: ~124 GB")

    # Download in batches
    batch_size = 50
    failed_batches = []

    for i in range(0, total_series, batch_size):
        batch_num = i // batch_size + 1
        total_batches = (total_series + batch_size - 1) // batch_size

        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"Processing batch {batch_num}/{total_batches} (series {i + 1}-{min(i + batch_size, total_series)})")
        logger.info(f"{'=' * 60}")

        batch = series_data.iloc[i:i + batch_size]

        if not download_with_retry(batch):
            failed_batches.append(batch_num)
            logger.warning(f"Batch {batch_num} had failures, continuing...")

        # Small delay between batches
        if i + batch_size < total_series:
            logger.info("Waiting 10 seconds before next batch...")
            time.sleep(10)

    logger.info("=" * 60)
    logger.info("Download process completed!")
    logger.info(f"End time: {datetime.now()}")

    if failed_batches:
        logger.warning(f"Batches with failures: {failed_batches}")
        logger.info("You can re-run the script to retry failed downloads")
    else:
        logger.info("All batches completed successfully!")

    logger.info("=" * 60)

except Exception as e:
    logger.error(f"Fatal error during download: {str(e)}")
    logger.exception("Full traceback:")
    raise

