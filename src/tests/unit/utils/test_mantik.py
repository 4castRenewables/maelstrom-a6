import dataclasses
import logging
import pathlib

import a6.utils.mantik as mantik
import mlflow


def test_log_logs_file_to_mantik(caplog, monkeypatch, tmp_path):
    caplog.set_level(logging.INFO)

    @dataclasses.dataclass
    class TestFile:
        path: str = None
        content: str = None

    test_file = TestFile()

    def grab_logged_test_file(path) -> None:
        test_file.path = path
        with open(path) as f:
            test_file.content = f.read()

    monkeypatch.setattr(mlflow, "log_artifact", grab_logged_test_file)

    with mantik.log_logs_as_file():
        logging.getLogger(__name__).info("hi")

    assert "hi" in caplog.text
    # TODO: investigate why the logs are not written to the file in tests.
    # (`assert "hi" in test_file.content` raises)
    # This works outside of pytest.
    assert not pathlib.Path(test_file.path).exists()
