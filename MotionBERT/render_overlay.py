import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from tqdm import tqdm
import os

def render_frame_3d(pose_3d):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=80)
    
    # Set limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    # Draw bones
    bones = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
             [0, 7], [7, 8], [8, 9], [8, 11], [11, 12], [12, 13],
             [8, 14], [14, 15], [15, 16], [9, 10]]
    
    for i, j in bones:
        xs = [-pose_3d[i, 0], -pose_3d[j, 0]]
        ys = [-pose_3d[i, 2], -pose_3d[j, 2]]
        zs = [-pose_3d[i, 1], -pose_3d[j, 1]]
        ax.plot(xs, ys, zs, c='b')
    
    # Hide axes
    ax.axis('off')
    
    # Render to image
    fig.canvas.draw()
    
    # Use tostring_argb instead of tostring_rgb
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert ARGB to RGB
    img = img[:, :, 1:4]
    
    plt.close(fig)
    return img

def combine_frames(left, right):
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + right.shape[1]
    combined = np.zeros((height, width, 3), dtype=np.uint8)
    combined[:left.shape[0], :left.shape[1]] = left
    combined[:right.shape[0], left.shape[1]:] = right
    return combined

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='Path to 3D keypoints .npy file')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Path to output rendered video')
    args = parser.parse_args()

    keypoints_3d = np.load(args.pred)  # (T, 17, 3)
    reader = imageio.get_reader(args.video)
    fps = reader.get_meta_data()['fps']

    output_writer = imageio.get_writer(args.output, fps=fps)

    for i, frame in tqdm(enumerate(reader), total=keypoints_3d.shape[0]):
        if i >= keypoints_3d.shape[0]:
            break
        pose_frame = render_frame_3d(keypoints_3d[i])  # 3D pose
        pose_frame = cv2.resize(pose_frame, (frame.shape[1], frame.shape[0]))
        combined = combine_frames(frame, pose_frame)
        output_writer.append_data(combined)

    output_writer.close()
    print(f"✅ Overlay video saved to {args.output}")

if __name__ == '__main__':
    main()