import a6


if __name__ == "__main__":
    parser = a6.cli.training.create_hdbscan_parser()
    args = parser.parse_args()

    data = a6.cli.data.read_ecmwf_ifs_hres_data(
        path=args.data,
        level=args.level,
    )

    hyperparameters = a6.studies.HyperParameters(
        n_components_start=args.n_components_start,
        n_components_end=args.n_components_end,
        min_cluster_size_start=args.min_cluster_size_start,
        min_cluster_size_end=args.min_cluster_size_end,
    )

    clusters = a6.studies.perform_pca_and_hdbscan_hyperparameter_study(
        data=data,
        vary_data_variables=args.vary_data_variables,
        hyperparameters=hyperparameters,
        use_varimax=args.use_varimax,
        log_to_mantik=args.log_to_mantik,
    )
