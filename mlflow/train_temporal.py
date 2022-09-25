import a6


if __name__ == "__main__":
    parser = a6.cli.training.create_temporal_parser()
    args = parser.parse_args()

    data = a6.cli.data.read_ecmwf_ifs_hres_data(
        path=args.data,
        level=args.level,
    )

    clusters = a6.studies.perform_temporal_range_study(
        data=data,
        n_components=args.n_components,
        min_cluster_size=args.min_cluster_size,
        use_varimax=args.use_varimax,
        log_to_mantik=args.log_to_mantik,
    )
