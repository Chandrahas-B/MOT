import numpy as np
import cv2
from . import colormaps

def bboxes(frame, results, detector):

    frame = np.array(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(round(i, 2)) for i in box.tolist()]
        # print(
        #         f"Detected {detector.model.config.id2label[label.item()]} with confidence "
        #         f"{round(score.item(), 3)} at location {box}"
        # )
        class_name = detector.model.config.id2label[label.item()]
        xmin, ymin, xmax, ymax = box
        color = colormaps.get(label.item(), (255, 255, 255))
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        cv2.putText(frame, f"{class_name}-{score.item():.2f}", (xmin-2, ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            
    return frame