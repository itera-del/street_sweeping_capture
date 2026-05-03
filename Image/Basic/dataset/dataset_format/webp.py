from PIL import Image
import imageio

# 打开 WebP 图片
images = [Image.open('/mnt/huanyuan/temp/video/overview_jpg/overview_wps图片_1.png' % i) for i in range(1, 10)]

# 保存为 GIF
imageio.mimsave('/mnt/huanyuan/temp/video/overview.gif', images, duration=0.5)