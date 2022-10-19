from pynvml import *  # noqa: F403

from Hive.utils.log_utils import get_logger

try:
    nvmlInit()  # noqa: F405
    h = nvmlDeviceGetHandleByIndex(0)  # noqa: F405
    info = nvmlDeviceGetMemoryInfo(h)  # noqa: F405

    threshold_gb = 8

    available_gb = info.free / (1024 * 1024 * 1024)

    GPU_AVAILABLE = available_gb >= threshold_gb


except:  # noqa: E722
    GPU_AVAILABLE = False

logger = get_logger("Hive", "INFO")
logger.info("Hive GPU Available: {}".format(GPU_AVAILABLE))
os.environ["GPU_AVAILABLE"] = str(GPU_AVAILABLE)  # noqa: F405
