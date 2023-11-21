python

class YOLOv5Detector:
    def __init__(self, weights, data, device='', half=False, dnn=False):
        self.weights = weights
        self.data = data
        self.device = device
        self.half = half
        self.dnn = dnn
        self.model = None
        self.stride = None
        self.names = None
        self.pt = None
        self.jit = None
        self.onnx = None
        self.engine = None

    def load_model(self):
        # Load model
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        )

        # Half
        half = self.half and (self.pt or self.jit or self.onnx or self.engine) and device.type != 'cpu'
        if self.pt or self.jit:
            self.model.model.half() if half else self.model.model.float()

    def run(self, img, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, max_det=1000, classes=None,
            agnostic_nms=False, augment=False):
        cal_detect = []

        device = select_device(self.device)
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

        # Set Dataloader
        im = letterbox(img, imgsz, self.stride, self.pt)[0]

        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=augment)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]}'
                    cal_detect.append([label, xyxy])
        return cal_detect


def detect():
    weights = ROOT / 'best.pt'
    data = ROOT / 'data/coco128.yaml'
    device = ''
    half = False
    dnn = False

    detector = YOLOv5Detector(weights, data, device, half, dnn)
    detector.load_model()

    image = cv2.imread("./images/1.jpg")
    results = detector.run(image)
    for i in results:
        box = i[1]
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        print(i[0])
        cv2.rectangle(image, p1, p2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.imshow('image', image)
    cv2.waitKey(0)


detect()
