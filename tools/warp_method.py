
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# 给定目标框的坐标
target_coords1 = [
    [
        [
            0.8515625,
            0.357177734375,
            0.30517578125,
            0.306884765625
        ]
    ]
]

target_coords2 = [
    [
        [
            0.5166015625,
            0.56787109375,
            0.420654296875,
            0.33056640625
        ]
    ]
]

target_coords3 = [
    [
        [
            0.86083984375,
            0.83544921875,
            0.286865234375,
            0.298095703125
        ]
    ]
]

target_coords4 = [
    [
        [
            0.61474609375,
            0.5126953125,
            0.2425537109375,
            0.273681640625
        ]
    ]
]
# 计算目标框的中心点坐标
def get_center_coords(coords):
    cx, cy, width, height = coords[0]
    return cx, cy

# 获取目标框的中心点坐标
x1_1, y1_1  = get_center_coords(target_coords1[0])
x2_1, y2_1 = get_center_coords(target_coords2[0])
x1_2, y1_2 = get_center_coords(target_coords3[0])
x2_2, y2_2  = get_center_coords(target_coords4[0])

# 计算方向向量
vector1 = np.array([x2_1 - x1_1, y2_1 - y1_1])  # 线1的方向向量
vector2 = np.array([x2_2 - x1_2, y2_2 - y1_2])  # 线2的方向向量

# 计算叉积（用于判断旋转方向）
cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

# 使用 dot product 计算夹角的余弦值
dot_product = np.dot(vector1, vector2)
magnitude1 = np.linalg.norm(vector1)
magnitude2 = np.linalg.norm(vector2)

# 计算夹角的余弦值
cos_angle = dot_product / (magnitude1 * magnitude2)

# 计算夹角（弧度）
angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 防止数值误差导致输入超出[-1, 1]
# 将弧度转换为角度
angle_degrees = np.degrees(angle_radians)

# 如果叉积为负，表示顺时针旋转，角度为负
if cross_product < 0:
    angle_degrees = -angle_degrees

print(f"两条线之间的夹角为: {angle_degrees:.2f}°")

# 可视化
fig, ax = plt.subplots()
ax.plot([x1_1, x2_1], [y1_1, y2_1], label="Line 1", color='r')
ax.plot([x1_2, x2_2], [y1_2, y2_2], label="Line 2", color='b')

ax.scatter([x1_1, x1_2], [y1_1, y1_2], color='black')  # 起点

ax.set_title(f"Angle between lines: {angle_degrees:.2f}°")
plt.legend()
plt.grid(True)
plt.show()


import cv2

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 动态计算新尺寸
    cos = abs(np.cos(np.deg2rad(angle)))
    sin = abs(np.sin(np.deg2rad(angle)))
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵的平移分量
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotation_matrix[0, 2] += (new_w - w) // 2
    rotation_matrix[1, 2] += (new_h - h) // 2

    # 执行旋转
    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h), borderValue=(255, 255, 255))
    return rotated_img
image_path = r"X:\_paper4\GeoText-1652-main\GeoText-1652-main\GeoText1652_Dataset\images\train\1573\image-23.jpeg"
# 打开图像
# image_path = "path_to_your_image.jpg"  # 替换为你的图像路径
image = cv2.imread(image_path)

# 旋转图像，注意 PIL 中的 rotate 是逆时针旋转
# rotated_image = image.rotate(-angle_degrees, expand=True)  # expand=True 防止裁剪

cv2.imwrite(r'X:\_paper4\GeoText-1652-main\GeoText-1652-main\GeoText1652_Dataset\images\test\222.jpg',rotate_image(image, -angle_degrees))
# cv2.imshow("1", rotate_image(image, -angle_degrees))
# 显示旋转后的图像
# plt.imshow(rotated_image)
# plt.title(f"Rotated by {angle_degrees}°")
# plt.axis('off')  # 关闭坐标轴
# plt.show()