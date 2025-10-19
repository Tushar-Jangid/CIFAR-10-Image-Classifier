import requests
import sys

def predict(image_path,url='D:\DP learning\Project2\dog.jpg'):
    with open (image_path, 'rb') as f:
        files = {"file":(image_path,f,"image/jpeg")}
        resp = requests.post(url,files=files)
        print(resp.status_code,resp.text)

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python perdict_client.py path/to/image")
    else:
        predict(sys.argv[1])