import numpy as np
import os


def image_to_pointcloud(image, step=1):
    """
    将深度图像转换为点云坐标
    image: (H, W) 形状的深度图
    step: 采样步长，用于降采样（如果点太多）
    returns: (N, 3) 形状的点云，包含 (x, y, depth)
    """
    H, W = image.shape

    # 创建网格坐标
    x = np.arange(0, W, step)
    y = np.arange(0, H, step)
    xx, yy = np.meshgrid(x, y)

    # 采样深度值
    depth = image[::step, ::step].flatten()

    # 组合成点云 (x, y, depth)
    points = np.stack([xx.flatten(), yy.flatten(), depth], axis=1)

    return points


def scale_pointcloud(points, sx, sy, sz):
    return points * np.array([sx, sy, sz])


def translate_pointcloud(points, tx, ty, tz):
    return points + np.array([tx, ty, tz])


def rotate_pointcloud(points, axis, angle):
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
    else:
        raise ValueError("轴必须是'x'、'y'或'z'")
    return points @ R.T


def process_single_file(filepath, filename, output_folder, base_name):
    """处理单个深度图文件"""
    try:
        depth_map = np.load(filepath)
        print(f"原始形状: {depth_map.shape}")

        # 将深度图转换为点云（降采样以减少数据量）
        step = 4  # 每4个像素采样一个点，可根据需要调整
        points = image_to_pointcloud(depth_map, step=step)
        print(f"点云形状: {points.shape}")

    except Exception as e:
        print(f"加载文件 {filename} 失败：{e}")
        return False

    # 保存原始点云
    np.save(os.path.join(output_folder, f"{base_name}_原始点云.npy"), points)

    # 缩放
    scaled_1 = scale_pointcloud(points, 6, 6, 1)
    scaled_2 = scale_pointcloud(points, 0.6, 0.6, 1)
    np.save(os.path.join(output_folder, f"{base_name}_缩放1.npy"), scaled_1)
    np.save(os.path.join(output_folder, f"{base_name}_缩放2.npy"), scaled_2)

    # 平移
    translated_1 = translate_pointcloud(points, 1.0, 3.0, 0)
    translated_2 = translate_pointcloud(points, 0, 7.0, 5.0)
    np.save(os.path.join(output_folder, f"{base_name}_平移1.npy"), translated_1)
    np.save(os.path.join(output_folder, f"{base_name}_平移2.npy"), translated_2)

    # 旋转
    rotated_1 = rotate_pointcloud(translated_1, 'x', np.pi / 4)
    rotated_2 = rotate_pointcloud(translated_1, 'y', np.pi / 8)
    rotated_3 = rotate_pointcloud(translated_1, 'z', np.pi / 6)
    np.save(os.path.join(output_folder, f"{base_name}_旋转1.npy"), rotated_1)
    np.save(os.path.join(output_folder, f"{base_name}_旋转2.npy"), rotated_2)
    np.save(os.path.join(output_folder, f"{base_name}_旋转3.npy"), rotated_3)

    return True


def process_folder():
    input_base = "../depth_data/merged"
    output_abnormal = "data_expansion_1/abnormal"
    output_normal = "data_expansion_1/normal"

    os.makedirs(output_abnormal, exist_ok=True)
    os.makedirs(output_normal, exist_ok=True)

    # 处理 abnormal 文件夹
    abnormal_folder = os.path.join(input_base, "abnormal")
    if os.path.exists(abnormal_folder):
        abnormal_files = [f for f in os.listdir(abnormal_folder) if f.endswith('.npy')]
        print(f"在 '{abnormal_folder}' 中找到 {len(abnormal_files)} 个 .npy 文件")

        for filename in abnormal_files[:5]:  # 先测试5个文件
            filepath = os.path.join(abnormal_folder, filename)
            base_name = os.path.splitext(filename)[0]
            success = process_single_file(filepath, filename, output_abnormal, base_name)
            if success:
                print(f"处理完成：{filename}")
            else:
                print(f"处理失败：{filename}")
    else:
        print(f"警告：abnormal 文件夹不存在：{abnormal_folder}")

    # 处理 normal 文件夹
    normal_folder = os.path.join(input_base, "normal")
    if os.path.exists(normal_folder):
        normal_files = [f for f in os.listdir(normal_folder) if f.endswith('.npy')]
        print(f"在 '{normal_folder}' 中找到 {len(normal_files)} 个 .npy 文件")

        for filename in normal_files[:5]:  # 先测试5个文件
            filepath = os.path.join(normal_folder, filename)
            base_name = os.path.splitext(filename)[0]
            success = process_single_file(filepath, filename, output_normal, base_name)
            if success:
                print(f"处理完成：{filename}")
            else:
                print(f"处理失败：{filename}")
    else:
        print(f"警告：normal 文件夹不存在：{normal_folder}")

    print(f"\n处理完毕！")


if __name__ == "__main__":
    process_folder()