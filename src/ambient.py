import cv2
import numpy as np

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
GAUSS_SIGMA = 200
ALPHA_DIST = 50.0

IMG_PATH = "test_img3.png"
OUTPUT_PATH = "output3.png"

# TODO argparseでコマンド引数取れるようにする

print(f"DISPLAY SIZE: {DISPLAY_WIDTH} x {DISPLAY_HEIGHT}")
print("ALPHA_DIST: ", ALPHA_DIST)
print("GAUSS_SIGMA: ", GAUSS_SIGMA)
print("INPUT IMAGE PATH: ", IMG_PATH)
print("OUTPUT IMAGE PATH: ", OUTPUT_PATH)

org_img = cv2.imread(IMG_PATH)

# org_img = cv2.resize(org_img, (int(DISPLAY_WIDTH * 0.9), int(DISPLAY_HEIGHT * 0.9)))
height, width, _ = org_img.shape
print(f"INPUT IMAGE SIZE: {width} x {height}")

# * Create a transparent gradient
alpha_channels = np.zeros((height, width), dtype=np.uint16)
a_height, a_width = alpha_channels.shape
# ! too slow
for y in range(a_height):
    for x in range(a_width):
        alpha_channels[y, x] = int(255 * min(min(x, y), min(a_width - x, a_height - y)) / ALPHA_DIST)
alpha_channels = np.clip(alpha_channels, 0.0, 255.0)
# add alpha channel
add_alpha_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2BGRA)
add_alpha_img[:, :, 3] = alpha_channels

# * Padding a blur effect background
# calculation of padding area
pad_width = int((DISPLAY_WIDTH - width) / 2)
pad_height = int((DISPLAY_HEIGHT - height) / 2)
pad_width = pad_width if pad_width > 0 else 0
pad_height = pad_height if pad_height > 0 else 0

# TODO 画像サイズがディスプレイサイズを超過している場合は丸め込む処理をいれる
# re_height = 0
# re_width = 0
# if pad_width < 0:
#     aspect = width / height
#     re_height = int(1080 * aspect)
#     pad_width = 0
# if pad_height < 0:
#     aspect = height / width
#     re_width = int(1920 * aspect)
#     pad_height = 0

# if re_height > 0:
#     org_img = cv2.resize(org_img, (re_height, 1080))
# if re_width > 0:
#     org_img = cv2.resize(org_img, (1920, re_width))

# add padding to image
padding_img = cv2.copyMakeBorder(org_img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_REFLECT)
bler_img = cv2.GaussianBlur(padding_img, (0, 0), GAUSS_SIGMA)  # ! too slow

x1, y1, x2, y2 = pad_width, pad_height, width + pad_width, height + pad_height,
bler_img[y1:y2, x1:x2] = bler_img[y1:y2, x1:x2] * (1 - add_alpha_img[:, :, 3:] / 255) + add_alpha_img[:, :, :3] * (add_alpha_img[:, :, 3:] / 255)

# cv2.imshow("preview", bler_img)
# cv2.waitKey(0)
cv2.imwrite(OUTPUT_PATH, bler_img)
