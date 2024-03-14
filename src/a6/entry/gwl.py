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
    select_dwd_area: bool = True,
    testing: bool = True,
    log_to_mlflow: bool = True,
) -> None:
    device = utils.distributed.get_single_device()
    logger = utils.logging.create_logger(
        global_rank=0,
        local_rank=0,
        verbose=False,
    )

    logger.info("Initialized logging")

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

    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        batch_size=64 if not testing else 1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=64 if not testing else 1,
        num_workers=0,
        pin_memory=True,
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

    model = (
        # models.cnn.Model(
        #     in_channels=train_set.n_channels,
        #     n_classes=n_classes,
        #     example=train_set[0][0],
        # )
        models.resnet.resnet50w4(
            in_channels=train_set.n_channels,
            n_classes=n_classes,
        )
        if not testing
        else models.cnn.TestingModel(
            in_channels=train_set.n_channels,
            n_classes=n_classes,
            example=train_set[0][0],
        )
    )

    logger.info("%s", model)

    # Copy model to GPU
    model.to(device)

    training.train.train(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        log_to_mlflow=log_to_mlflow,
    )
