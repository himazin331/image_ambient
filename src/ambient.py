import cv2
import argparse as arg
import numpy as np

parser = arg.ArgumentParser(description="Ambient Image Generator")
parser.add_argument("--img_path", "-i", type=str, default=None, help="Input image path (PNG Only)")
parser.add_argument("--output_path", "-o", type=str, default="output.png", help="Output image path")
parser.add_argument("--disp_width", "-dw", type=int, default=1920, help="Display width (default: 1920)")
parser.add_argument("--disp_height", "-dh", type=int, default=1080, help="Display height (default: 1080)")
parser.add_argument("--gauss_sigma", "-gs", type=float, default=200.0, help="Gaussian blur sigma (default: 200)")
parser.add_argument("--alpha_dist", "-ad", type=float, default=50.0, help="Distance of transparent gradient of image border (default: 50.0)")
parser.add_argument("--pad_mod", "-pm", type=str, choices=["reflect", "constant"], default="reflect", help="Padding mode (default: reflect)")
args = parser.parse_args()

print(f"DISPLAY SIZE: {args.disp_width} x {args.disp_height}")
print("ALPHA_DIST: ", args.alpha_dist)
print("GAUSS_SIGMA: ", args.gauss_sigma)
print("INPUT IMAGE PATH: ", args.img_path)
print("OUTPUT IMAGE PATH: ", args.output_path)

org_img = cv2.imread(args.img_path)
height, width, _ = org_img.shape
print(f"INPUT IMAGE SIZE: {width} x {height}")

pad_width: int = int((args.disp_width - width) / 2)
pad_height: int = int((args.disp_height - height) / 2)
# * Rounding if image size exceeds display size
if pad_width < 0 or pad_height < 0:
    print("Image size is too large. Resizing...")
    if pad_width < 0:
        aspect: float = height / width
        re_height: int = int(args.disp_width * aspect)
        org_img = cv2.resize(org_img, (args.disp_width, re_height))
        pad_width = 0
    if pad_height < 0:
        aspect: float = width / height
        re_width: int = int(args.disp_height * aspect)
        org_img = cv2.resize(org_img, (re_width, args.disp_height))
        pad_height = 0
    height, width, _ = org_img.shape
    print(f"RESIZED IMAGE SIZE: {width} x {height}")
    pad_width = int((args.disp_width - width) / 2)
    pad_height = int((args.disp_height - height) / 2)

# * Create a transparent gradient
if args.alpha_dist > 1:
    alpha_channels = np.zeros((height, width), dtype=np.uint16)
    a_height, a_width = alpha_channels.shape
    for y in range(a_height):
        for x in range(a_width):
            alpha_channels[y, x] = int(255 * min(min(x, y), min(a_width - x, a_height - y)) / args.alpha_dist)
    alpha_channels = np.clip(alpha_channels, 0.0, 255.0)
    # add alpha channel
    add_alpha_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2BGRA)
    add_alpha_img[:, :, 3] = alpha_channels
else:
    add_alpha_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2BGRA)

# * Padding a blur effect background
# add padding to image
if args.pad_mod == "reflect":
    border_type: int = cv2.BORDER_REFLECT
else:
    border_type = cv2.BORDER_CONSTANT
padding_img = cv2.copyMakeBorder(org_img, pad_height, pad_height, pad_width, pad_width, border_type, value=(255, 255, 255))
padding_img = cv2.resize(padding_img, (args.disp_width, args.disp_height))  # Completely fill in just a few shortfalls.
bler_img = cv2.GaussianBlur(padding_img, (0, 0), args.gauss_sigma)  # ! too slow
# overwrite image
x1, y1, x2, y2 = pad_width, pad_height, width + pad_width, height + pad_height,
bler_img[y1:y2, x1:x2] = bler_img[y1:y2, x1:x2] * (1 - add_alpha_img[:, :, 3:] / 255) + add_alpha_img[:, :, :3] * (add_alpha_img[:, :, 3:] / 255)

height, width, _ = bler_img.shape
print(f"OUTPUT IMAGE SIZE: {width} x {height}")

# cv2.imshow("preview", bler_img)
# cv2.waitKey(0)
cv2.imwrite(args.output_path, bler_img)
