# definitions
from settings import MODE_DEBUG
from settings import MODE_FORCE_UPDATE
from settings import PREDICT
from settings import MODEL_TFLITE

MODE_DEBUG_PROD = False # default = False 
MODE_FORCE_UPDATE_PROD = False # default = False 
PREDICT_PROD = True # default = True 
MODEL_TFLITE_PROD = True # default = True 

# TESTS
class TestSettings:
    def test_set_prod(self):
        assert MODE_DEBUG == MODE_DEBUG_PROD
        assert MODE_FORCE_UPDATE == MODE_FORCE_UPDATE_PROD
        assert PREDICT == PREDICT_PROD
        assert MODEL_TFLITE == MODEL_TFLITE_PROD