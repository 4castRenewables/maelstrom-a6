import logging
import sys
from concurrent import futures
from pathlib import Path

import cdsapi

logger = logging.getLogger(__name__)

DATASET = "reanalysis-era5-pressure-levels"


def _create_request_body(year: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            # "relative_humidity",
            # "temperature",
            # "u_component_of_wind",
            # "v_component_of_wind",
        ],
        "year": [str(year)],
        "month": [f"{i:02d}" for i in range(1, 13)],
        "day": [f"{i:02d}" for i in range(1, 32)],
        "time": ["12:00"],
        "pressure_level": ["500"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [70, -70, 25, 50],
    }


def _download_data(requests: dict[Path, dict], client: cdsapi.Client) -> None:
    success = True
    n_downloads_failed = 0
    n_files = len(requests)

    logger.info("Downloading %s files...", n_files)

    logger.debug("Creating thread pool for download with %s workers", n_files)

    with futures.ThreadPoolExecutor(max_workers=n_files) as executor:
        future_to_local_path = {
            executor.submit(
                client.retrieve,
                name=DATASET,
                request=request,
                target=path,
            ): path
            for path, request in requests.items()
        }

        logger.info("All file downloads initiated...")

        while future_to_local_path:
            done, _ = futures.wait(
                future_to_local_path, return_when=futures.FIRST_COMPLETED
            )

            for future in done:
                path = future_to_local_path.pop(future)
                try:
                    result = future.result()
                except Exception:  # noqa: B902
                    logger.warning(
                        "Download of %s has failed: %s",
                        path.as_posix(),
                        exc_info=True,
                    )
                    success = False
                    n_downloads_failed += 1
                else:
                    logger.info(
                        "Download of %s succeeded: %s", path.as_posix(), result
                    )

    if success:
        logger.info("Download of all files succeeded!")
    else:
        logger.error(
            "Download of %s of a total of %s files failed!",
            n_downloads_failed,
            n_files,
        )


if __name__ == "__main__":
    base = Path(sys.argv[1])
    client = cdsapi.Client()
    requests = {
        base / f"era5-{year}-12-UTC-500-hPa-geopotential.nc": _create_request_body(year)
        for year in range(1964, 2024)
    }

    _download_data(requests=requests, client=client)
