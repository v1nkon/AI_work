import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'E://study/soft/tesseract_ocr/tesseract.exe'
text = pytesseract.image_to_string(Image.open('data/captcha_denoising/0A7W.jpg'))
# with open('2.py', 'wb') as f:
#     f.write(text.encode('utf8'))
print(text )