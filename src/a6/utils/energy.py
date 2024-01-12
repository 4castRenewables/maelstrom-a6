import time
from multiprocessing import Event
from multiprocessing import Process
from multiprocessing import Queue

import pandas as pd


def get_energy_profiler(hardware_name):
    if hardware_name == "nvidia":
        return GetNVIDIAPower
    elif hardware_name == "amd":
        return GetAMDPower
    else:
        raise NotImplementedError(f"Unknown hardware_name {hardware_name}")


# NVIDIA GPUS
class GetNVIDIAPower:
    def __enter__(self):
        self.end_event = Event()
        self.power_queue = Queue()

        interval = 100  # ms
        self.smip = Process(
            target=self._power_loop,
            args=(self.power_queue, self.end_event, interval),
        )
        self.smip.start()
        return self

    def _power_loop(self, queue, event, interval):
        import pynvml as pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        device_list = [
            pynvml.nvmlDeviceGetHandleByIndex(idx)
            for idx in range(device_count)
        ]
        power_value_dict = {idx: [] for idx in range(device_count)}
        power_value_dict["timestamps"] = []
        last_timestamp = time.time()

        while not event.is_set():
            for idx, handle in enumerate(device_list):
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_value_dict[idx].append(power * 1e-3)
            timestamp = time.time()
            power_value_dict["timestamps"].append(timestamp)
            wait_for = max(0, 1e-3 * interval - (timestamp - last_timestamp))
            time.sleep(wait_for)
            last_timestamp = timestamp
        queue.put(power_value_dict)

    def __exit__(self, type, value, traceback):
        self.end_event.set()
        power_value_dict = self.power_queue.get()
        self.smip.join()

        self.df = pd.DataFrame(power_value_dict)

    def energy(self):
        _energy = []
        energy_df = (
            self.df.loc[:, self.df.columns != "timestamps"]
            .astype(float)
            .multiply(self.df["timestamps"].diff(), axis="index")
            / 3600
        )
        _energy = energy_df[1:].sum(axis=0).values.tolist()
        return _energy


# AMD GPUS


class GetAMDPower:
    def __enter__(self):
        self.end_event = Event()
        self.power_queue = Queue()

        interval = 100  # ms
        self.smip = Process(
            target=self._power_loop,
            args=(self.power_queue, self.end_event, interval),
        )
        self.smip.start()
        return self

    def _power_loop(self, queue, event, interval):
        import rsmiBindings as rmsi  # noqa: N813

        ret = rmsi.rocmsmi.rsmi_init(0)
        if rmsi.rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
            raise RuntimeError("Failed initializing rocm_smi library")
        device_count = rmsi.c_uint32(0)
        ret = rmsi.rocmsmi.rsmi_num_monitor_devices(rmsi.byref(device_count))
        if rmsi.rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
            raise RuntimeError("Failed enumerating ROCm devices")
        device_list = list(range(device_count.value))
        power_value_dict = {id: [] for id in device_list}
        power_value_dict["timestamps"] = []
        last_timestamp = time.time()
        start_energy_list = []
        for id in device_list:
            energy = rmsi.c_uint64()
            energy_timestamp = rmsi.c_uint64()
            energy_resolution = rmsi.c_float()
            ret = rmsi.rocmsmi.rsmi_dev_energy_count_get(
                id,
                rmsi.byref(energy),
                rmsi.byref(energy_resolution),
                rmsi.byref(energy_timestamp),
            )
            if rmsi.rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
                raise RuntimeError(f"Failed getting Power of device {id}")
            start_energy_list.append(
                round(energy.value * energy_resolution.value, 2)
            )  # unit is uJ

        while not event.is_set():
            for id in device_list:
                power = rmsi.c_uint32()
                ret = rmsi.rocmsmi.rsmi_dev_power_ave_get(
                    id, 0, rmsi.byref(power)
                )
                if rmsi.rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
                    raise RuntimeError(f"Failed getting Power of device {id}")
                power_value_dict[id].append(power.value * 1e-6)  # value is uW
            timestamp = time.time()
            power_value_dict["timestamps"].append(timestamp)
            wait_for = max(0, 1e-3 * interval - (timestamp - last_timestamp))
            time.sleep(wait_for)
            last_timestamp = timestamp

        energy_list = [0.0 for _ in device_list]
        for id in device_list:
            energy = rmsi.c_uint64()
            energy_timestamp = rmsi.c_uint64()
            energy_resolution = rmsi.c_float()
            ret = rmsi.rocmsmi.rsmi_dev_energy_count_get(
                id,
                rmsi.byref(energy),
                rmsi.byref(energy_resolution),
                rmsi.byref(energy_timestamp),
            )
            if rmsi.rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
                raise RuntimeError(f"Failed getting Power of device {id}")
            energy_list[id] = (
                round(energy.value * energy_resolution.value, 2)
                - start_energy_list[id]
            )

        energy_list = [
            (energy * 1e-6) / 3600 for energy in energy_list
        ]  # convert uJ to Wh
        queue.put(power_value_dict)
        queue.put(energy_list)

    def __exit__(self, type, value, traceback):
        self.end_event.set()
        power_value_dict = self.power_queue.get()
        self.energy_list_counter = self.power_queue.get()
        self.smip.join()

        self.df = pd.DataFrame(power_value_dict)

    def energy(self):
        _energy = []
        energy_df = (
            self.df.loc[:, self.df.columns != "timestamps"]
            .astype(float)
            .multiply(self.df["timestamps"].diff(), axis="index")
            / 3600
        )
        _energy = energy_df[1:].sum(axis=0).values.tolist()
        return _energy, self.energy_list_counter
