from ultralytics import YOLO 
import numpy as np
import cv2
import onnxruntime as ort 
import os
 
# Set global ONNX Runtime options
ort.set_default_logger_severity(3)  #

class YOLODetection:

    def __init__(self):

        self.model = YOLO('/home/rishabh/Desktop/yoloe-2.pt',task='segment')
        self.threshold_dice = 0.15
        self.threshold_iou = 0.15
        self.masks_data=[]
        self.classes_detected = []
        self.model.overrides['verbose'] = False
        self.model.overrides['device'] = '0'

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()

    def mask_dice(self,mask1,mask2):
         
        intersection = np.logical_and(mask1, mask2).sum()
        return 2* intersection / (mask1.sum() + mask2.sum())
    
    def mask_iou(self,mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union != 0 else 0.0
 
     
    def detection(self, frame):
        results = self.model.predict(source=frame, stream=True, show=False, conf=0.25, imgsz=640, verbose=False,
                                      device='0', max_det=50,
                                      agnostic_nms=True,
                                      half=True,
                                      augment=False,)

        result = list(results)[0]
        frame = result.orig_img
        h1, w1 = frame.shape[:2]
        area_total = h1 * w1
        boxes = result.boxes
        return_array = []

        if boxes is not None and result.masks is not None:
            boxes = list(boxes)
            masks = result.masks.data.cpu().numpy()
            detections = []

            if len(boxes) != 0 and len(masks) != 0:
                for box, mask in zip(boxes, result.masks):
                    mask_array = mask.data.cpu().numpy()
                    class_index = int(box.cls)
                    class_name = result.names[class_index].lower()
                    if class_name in ["desert", "sky", "drought", "sand", "fly", "kite"]:
                        continue
                    score = float(box.conf[0])
                    if score < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0:
                        continue
                    if x1 == w1 or x2 == w1 or y1 == h1 or y2 == h1:
                        continue
                    
                    # ✅ Calculate original bbox area (without padding) for thresholding
                    original_width = x2 - x1
                    original_height = y2 - y1
                    original_area = original_width * original_height
                    area_ratio = original_area / area_total
                    
                    min_area_threshold = 0.008 # 0.8% of image
                    max_area_threshold = 0.40  # 40% of image
                    
                    if area_ratio < min_area_threshold:
                        print(f"Skipped {class_name}: too small ({area_ratio:.3f} < {min_area_threshold})")
                        continue
                    if area_ratio > max_area_threshold:
                        print(f"Skipped {class_name}: too large ({area_ratio:.3f} > {max_area_threshold})")
                        continue
                    
                    print(f"✅ {class_name}: area ratio {area_ratio:.3f} (within {min_area_threshold}-{max_area_threshold})")
                    
                    # ADD PADDING with boundary checks (for ArUco detection only)
                    padding = 20  # pixels
                    x1_padded = max(0, x1 - padding)
                    y1_padded = max(0, y1 - padding)
                    x2_padded = min(w1, x2 + padding)
                    y2_padded = min(h1, y2 + padding)
                    
                    cropped = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                    
                    # Ensure cropped region is valid
                    if cropped.size == 0:
                        continue
                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
                    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
                    if ids is not None and len(ids) > 0:
                        print("Detected ArUco markers!")
                        continue
                    
                    # Store all valid detections for this frame
                    detections.append({
                        "class_name": class_name,
                        "class_index": class_index,
                        "bbox": (x1, y1, x2, y2),  # Original bbox coordinates
                        "mask": mask_array,
                        "area": original_area,  # ✅ Use original area for NMS comparison
                        "area_ratio": area_ratio  # Store for debugging
                    })

                 
                keep = [True] * len(detections)
                for i in range(len(detections)):
                    for j in range(i + 1, len(detections)):
                        iou = self.mask_iou(detections[i]["mask"], detections[j]["mask"])
                        if iou > self.threshold_iou:
                            
                            if detections[i]["area"] >= detections[j]["area"]:
                                keep[j] = False
                            else:
                                keep[i] = False

                for idx, det in enumerate(detections):
                    if keep[idx]:
                        return_array.append((det["class_name"], det["class_index"], det["bbox"],masks))

        return return_array

