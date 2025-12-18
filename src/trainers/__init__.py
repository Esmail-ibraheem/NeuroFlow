"""
Compatibility shim so that imports like `import trainers.clm`
resolve to the actual `autotrain.trainers` package.
"""

import sys

from autotrain import trainers as _autotrain_trainers

sys.modules[__name__] = _autotrain_trainers
