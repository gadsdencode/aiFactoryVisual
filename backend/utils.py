# ==============================================================================
#  File: utils.py
#  - Contains helper functions, including environment setup.
# ==============================================================================
import logging
import importlib.util
import platform
import torch

logger = logging.getLogger(__name__)


def is_bitsandbytes_available() -> bool:
    """Check if the bitsandbytes library is installed."""
    return importlib.util.find_spec("bitsandbytes") is not None


class Environment:
    """A container for the detected hardware and software environment."""
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.bnb_available = is_bitsandbytes_available()
        self.bf16_supported = self.cuda_available and torch.cuda.is_bf16_supported()
        # On Windows, prefer float16 for wider library compatibility (e.g., bitsandbytes)
        if platform.system() == "Windows":
            self.compute_dtype = torch.float16
        else:
            self.compute_dtype = torch.bfloat16 if self.bf16_supported else torch.float16
        self.device_name = torch.cuda.get_device_name(0) if self.cuda_available else "CPU"

    def setup_backends(self):
        """Configure PyTorch backends for optimal performance."""
        if self.cuda_available:
            logger.info(f"CUDA is available. Using device: {self.device_name}")
            # torch.cuda.get_device_capability returns a (major, minor) tuple
            major, minor = torch.cuda.get_device_capability(0)
            if (major, minor) >= (8, 0):
                logger.info("Enabling TF32 for Ampere+ GPUs for improved performance.")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")