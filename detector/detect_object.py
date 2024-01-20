from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from torch import nn

class DetectionTransformer(nn.Module):
    
    def __init__(self, process_path =  "models/pre-processor", model_path = "models/DeTr", **kwargs):
        super().__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = DetrImageProcessor.from_pretrained(process_path, revision="no_timm")
        
        self.model = DetrForObjectDetection.from_pretrained(model_path, revision="no_timm")
        self.model.to(self.device).eval()
        
    def predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        with torch.no_grad():         
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold= 0.80)[0]
        
        return results
            
            
if __name__ == '__main__':
    object_detector = DetectionTransformer()
    from PIL import Image
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    results = object_detector.predict(image)
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                    f"Detected {object_detector.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
            )