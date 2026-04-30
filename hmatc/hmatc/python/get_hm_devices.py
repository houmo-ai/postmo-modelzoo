import hmatc.python.smi as smi
import os

def get_hm_devices(ndevices=1) -> list:
    """
    Get a list of HM devices available on the system.

    Returns:
        List[HMDevice]: A list of HMDevice objects representing the available HM devices.
    """
    DEFAULT_RUN_DEVEVICES = [i for i in range(ndevices)]
    if not os.getenv("HOUMO_VISIBLE_DEVICES"):
        assert smi.device_ctc_check(DEFAULT_RUN_DEVEVICES)
        return DEFAULT_RUN_DEVEVICES

    env_devices = os.getenv("HOUMO_VISIBLE_DEVICES").split(",")
    env_devices = [int(dev.strip()) for dev in env_devices]
    assert len(env_devices) >= ndevices, f"Not enough devices specified in HOUMO_VISIBLE_DEVICES. Required: {ndevices}, Provided: {len(env_devices)}"

    dev_start_idx = sorted(env_devices)[0]

    DEFAULT_RUN_DEVEVICES = [dev_start_idx + i for i in range(ndevices)]
    assert smi.device_ctc_check(DEFAULT_RUN_DEVEVICES)
    return DEFAULT_RUN_DEVEVICES
