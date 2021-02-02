import classla
from tests_classla import *

classla.download('sl', dir=TEST_MODELS_DIR)
classla.download('sl', package='ssj_jos', dir=TEST_MODELS_DIR)
classla.download('sl', package='nonstandard', dir=TEST_MODELS_DIR)
classla.download('hr', dir=TEST_MODELS_DIR)
classla.download('hr', package='nonstandard', dir=TEST_MODELS_DIR)
classla.download('sr', dir=TEST_MODELS_DIR)
classla.download('sr', package='nonstandard', dir=TEST_MODELS_DIR)
classla.download('bg', dir=TEST_MODELS_DIR)
classla.download('mk', dir=TEST_MODELS_DIR)
