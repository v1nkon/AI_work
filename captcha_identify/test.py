import numpy as np
from PIL import Image

# 5.790652513504028

def testClose():
    f1 = open('data/captcha/0A7W.jpg', 'rb')
    img = Image.open(f1)
    arr = np.array(img)
    f1.close()
    pic = Image.fromarray(arr)
    pic_l = pic.convert('L')
    pic_l.show()

def testClose2():
    f1 = open('data/captcha/0A7W.jpg', 'rb')
    img = Image.open(f1)
    f1.close()
    pic_l = img.convert('L')
    pic_l.show()

def testClose3():
    f1 = Image.open( 'data/captcha/0A7W.jpg' )
    f1.close()

# testClose()
# testClose2()

testClose3()


W2EW.jpg
W8VW.jpg
RV0W.jpg
ZOZW.jpg
ZQOW.jpg
X0UM.jpg
WSAW.jpg
WV1W.jpg
VPJW.jpg