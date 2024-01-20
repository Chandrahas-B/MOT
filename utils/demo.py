import requests

API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("/home/chandrahas/Pictures/Screenshots/Screenshot from 2024-01-15 00-34-47.png")
print(output)