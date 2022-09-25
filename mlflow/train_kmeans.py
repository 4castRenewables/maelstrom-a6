import itertools

import a6

if __name__ == "__main__":
    parser = a6.cli.training.create_kmeans_parser()
    args = parser.parse_args()

    data = a6.cli.data.read_ecmwf_ifs_hres_data(
        path=args.data,
        level=args.level,
    )

    parameters = itertools.product(
        args.n_components, args.n_clusters, args.use_varimax
    )

    for n_components, n_clusters, use_varimax in parameters:
        a6.studies.perform_pca_and_kmeans(
            data=data,
            n_components=n_components,
            n_clusters=n_clusters,
            use_varimax=use_varimax,
            log_to_mantik=args.log_to_mantik,
        )
