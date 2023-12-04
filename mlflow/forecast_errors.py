import logging

import a6


if __name__ == "__main__":
    a6.utils.logging.log_to_stdout(level=logging.DEBUG)
    a6.evaluation.forecast.simulate_forecast_errors()
