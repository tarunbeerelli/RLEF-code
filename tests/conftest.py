import sys
from unittest.mock import MagicMock

# ─── MOCK HEAVY FRAMEWORKS IF NOT INSTALLED (CI SANITY CHECK) ────────────────
try:
    import vllm  # noqa: F401
except ImportError:
    # Stub out vllm components so imports don't explode in raw CPU environments
    mock_vllm = MagicMock()
    mock_vllm.AsyncEngineArgs = MagicMock()
    mock_vllm.AsyncLLMEngine = MagicMock()
    mock_vllm.SamplingParams = MagicMock()
    sys.modules["vllm"] = mock_vllm
    sys.modules["vllm.lora.request"] = MagicMock()

try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()

try:
    import peft  # noqa: F401
except ImportError:
    sys.modules["peft"] = MagicMock()

try:
    import wandb  # noqa: F401
except ImportError:
    sys.modules["wandb"] = MagicMock()