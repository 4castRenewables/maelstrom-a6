import pathlib

import torch.utils.data
import xarray as xr

import a6.datasets as datasets
import a6.models as models
import a6.training as training
import a6.utils as utils


def main(
    epochs: int,
    data_path: pathlib.Path,
    gwl_path: pathlib.Path,
    architecture: str,
    select_dwd_area: bool = True,
    testing: bool = True,
    log_to_mlflow: bool = True,
) -> None:
    with utils.distributed.setup() as properties:
        coordinates = datasets.coordinates.Coordinates()

        ds = datasets.dwd.get_dwd_era5_data(
            path=data_path,
            pattern="*.nc",
            parallel_loading=False,
            select_dwd_area=select_dwd_area,
        ).sel(level=[500, 950])
        gwl = xr.open_dataset(gwl_path)

        start, end = gwl[coordinates.time][0], gwl[coordinates.time][-1]

        ds = ds.sel({coordinates.time: slice(start, end)})

        size = len(ds[coordinates.time])
        random_indexes = torch.randperm(size)
        train_size = int(0.8 * size)
        train_indexes = random_indexes[:train_size]
        test_indexes = random_indexes[train_size:]

        variables = datasets.variables.GWL()
        n_classes = int(gwl[variables.gwl].max())

        train_set = datasets.torch.xarray.WithGWLTarget(
            data_path=data_path,
            weather_dataset=ds.isel({coordinates.time: train_indexes}),
            gwl_dataset=gwl,
        )

        test_set = datasets.torch.xarray.WithGWLTarget(
            data_path=data_path,
            weather_dataset=ds.isel({coordinates.time: test_indexes}),
            gwl_dataset=gwl,
        )

        train_loader = utils.distributed.prepare_dataloader(
            train_set,
            batch_size=64 if not testing else 1,
            drop_last=False,
            properties=properties,
        )
        test_loader = utils.distributed.prepare_dataloader(
            test_set,
            batch_size=64 if not testing else 1,
            properties=properties,
        )
        logger.info(
            (
                "Building data done with %s/%s train/test images loaded, "
                "training on %s batches"
            ),
            len(train_set),
            len(test_set),
            len(train_loader),
        )

        if testing:
            model = models.cnn.TestingModel(
                in_channels=train_set.n_channels,
                n_classes=n_classes,
                example=train_set[0][0],
            )
            has_batchnorm = False
        elif architecture == "cnn":
            model = models.cnn.Model(
                in_channels=train_set.n_channels,
                n_classes=n_classes,
                example=train_set[0][0],
            )
            has_batchnorm = False
        else:
            model = models.resnet.Models[models.resnet.Architecture(architecture)](
                in_channels=train_set.n_channels,
                n_classes=n_classes,
            )
            has_batchnorm = True

        utils.distributed.prepare_model(
            model,
            has_batchnorm=has_batchnorm,
            properties=properties,
        )

        training.train.train(
            model=model,
            epochs=epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            log_to_mlflow=log_to_mlflow,
        )
