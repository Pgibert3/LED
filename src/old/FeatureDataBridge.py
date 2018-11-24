#Pickling port audio error is a result of not checking if __name__ == "__main__"
import sys
from FeatureExtractor import FeatureExtractor
from InputInterface import InputInterface
from Exceptions import DataBridgeException, AudioStreamException

class FeatureDataBridge:
    """Isolates Feature Extraction in a different process

	Attributes:

	"""
    def __init__(self):
        self.ext = FeatureExtractor()
        self.stream = None

    #Open the data bridge
    def open_bridge(self):
        try:
            self.ext.open_audio_stream()
        except AudioStreamException as e:
            print(e)
            sys.exit()

    #Close the data bridge
    def close_bridge(self):
        try:
            self.ext.close_audio_stream()
        except AudioStreamException as e:
            print(e)
            sys.exit()

    #Returns feature data from audio extraction process
    def get_feature_data(self):
        try:
            return self.ext.fetch_data()
        except AudioStreamException as e:
            print(e)
            sys.exit()
