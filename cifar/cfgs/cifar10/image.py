from PIL import Image
# 创建一个简单的图像
im = Image.new('RGB', (30, 30),)
# 保存这个图像
im.save('red.png')