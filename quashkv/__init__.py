"""quashkv — Near-optimal KV cache compression and vector quantization."""

__version__ = "0.1.0"

from .codebook import LloydMaxCodebook
from .quantizer import MSEQuantizer, InnerProductQuantizer
from .engine import QuashKVEngine
from .packing import pack_bits, unpack_bits
from .nn_search import QuashIndex
from .mixed_precision import MixedPrecisionMSEQuantizer, MixedPrecisionIPQuantizer

__all__ = [
    "LloydMaxCodebook",
    "MSEQuantizer",
    "InnerProductQuantizer",
    "QuashKVEngine",
    "QuashIndex",
    "MixedPrecisionMSEQuantizer",
    "MixedPrecisionIPQuantizer",
    "pack_bits",
    "unpack_bits",
]
