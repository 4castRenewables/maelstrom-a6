import pathlib

import a6
import mantik

if __name__ == "__main__":
    path = pathlib.Path(__file__).parent

    parser = a6.cli.compute_backend.create_parser()
    args = parser.parse_args()

    experiment_id = (
        args.experiment_id
        or a6.cli.compute_backend.read_experiment_id_from_env()
    )

    client = mantik.ComputeBackendClient.from_env()
    response = client.submit_run(
        experiment_id=experiment_id,
        entry_point=args.entry_point,
        mlflow_parameters=dict(args.mlflow_parameters),
        mlproject_path=path,
        backend_config_path=args.backend_config,
    )

    print(response.content)
