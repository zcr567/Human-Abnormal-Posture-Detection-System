import cv2
import numpy as np
import pyrealsense2 as rs


pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)


def get_frame_np():
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    return color_image, depth_image


def get_frame_raw():
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    return color_frame.get_data(), depth_frame.get_data()


if __name__ == "__main__":
    try:
        while True:
            color_imag, depth_imag = get_frame_np()

            cv2.imshow('Color_Image', color_imag)

            depth_gray = cv2.convertScaleAbs(depth_imag, alpha=255 / 4000)
            cv2.imshow('Depth Image', depth_gray)

            if cv2.waitKey(1) & 0xFF == ord('g'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
