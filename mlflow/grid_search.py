import a6
import xarray as xr


if __name__ == "__main__":
    parser = a6.cli.grid_search.create_parser()
    args = parser.parse_args()

    weather = a6.cli.data.read_ecmwf_ifs_hres_data(
        path=args.weather_data,
        level=args.level,
    )
    turbine = xr.open_dataset(args.turbine_data)

    gs = a6.studies.perform_forecast_model_grid_search(
        weather=weather,
        turbine=turbine,
        log_to_mantik=args.log_to_mantik,
    )
