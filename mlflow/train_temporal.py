import lifetimes


if __name__ == "__main__":
    parser = lifetimes.cli.training.create_temporal_parser()
    args = parser.parse_args()

    data = lifetimes.cli.data.read_ecmwf_ifs_hres_data(
        path=args.data,
        level=args.level,
    )

    clusters = lifetimes.studies.perform_temporal_range_study(
        data=data,
        n_components=args.n_components,
        min_cluster_size=args.min_cluster_size,
        use_varimax=args.use_varimax,
        log_to_mantik=args.log_to_mantik,
    )
