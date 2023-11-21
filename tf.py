python
class YOLOv5:
    def __init__(self, weights, imgsz=(640, 640)):
        self.weights = weights
        self.imgsz = imgsz
        self.model = self._build_model()

    def _build_model(self):
        # build the YOLOv5 model using the TensorFlow implementation
        # code here

    def detect(self, image):
        # run object detection on the input image
        # code here
