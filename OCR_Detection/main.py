import cv2
import numpy as np
import requests
import io
import json

img=cv2.imread("test3.png")
# cv2.imshow("img",img)
height,weight,_ =img.shape


roi=img[0:height,0:weight]

_,compresses_img=cv2.imencode('.png',roi,[1,90])
file_bytes=io.BytesIO(compresses_img)

url_api="https://api.ocr.space/parse/image"
result=requests.post(url_api,files={"test3.png":file_bytes},data={"apikey":"getYourOwn","language":"chs"})


result=result.content.decode()

result=json.loads(result)

# detected_text=result[0]
text_detected=result.get("ParsedResults")[0]
print(text_detected["ParsedText"])
# print(roi[0])

# print(img.shape)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()