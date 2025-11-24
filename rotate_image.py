import pygame
import numpy as np
import cv2
import math

# ---------------------------
# 3D 회전(임의 축) + 원근투영 + 퍼스펙티브 워핑
# ---------------------------

def rodrigues_rotate(points, axis_point, axis_dir, theta):
    """
    points: (N,3) 3D points
    axis_point: 축이 지나는 한 점 (3,)
    axis_dir: 축 방향 단위벡터 (3,)
    theta: 회전각(rad)
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 축 기준으로 평행이동
    p = points - axis_point

    # Rodrigues 공식
    k = axis_dir
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    p_rot = (p * cos_t +
             np.cross(k, p) * sin_t +
             k * (np.dot(p, k))[:, None] * (1 - cos_t))

    # 원래 위치로 복귀
    return p_rot + axis_point


def project_points(points3d, cam_dist=800.0, f=600.0, screen_center=(400, 300)):
    """
    간단한 pinhole 카메라 투영
    cam_dist: 카메라가 z축 양의 방향에서 떨어진 거리
    f: focal length(클수록 원근 약해짐)
    """
    cx, cy = screen_center
    x, y, z = points3d[:,0], points3d[:,1], points3d[:,2]

    # 카메라 앞쪽으로 이동된 z
    z_cam = z + cam_dist

    # 너무 뒤로 가면 발산하니 clamp
    z_cam = np.maximum(z_cam, 1e-3)

    x2d = f * (x / z_cam) + cx
    y2d = f * (y / z_cam) + cy
    return np.stack([x2d, y2d], axis=1)


def surf_to_cv(surf):
    """pygame.Surface -> cv2 BGR image"""
    arr = pygame.surfarray.array3d(surf)            # (w, h, 3)
    arr = np.transpose(arr, (1, 0, 2))             # (h, w, 3)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def cv_to_surf(img_bgr):
    """cv2 BGR image -> pygame.Surface"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = np.transpose(img_rgb, (1, 0, 2))     # (w, h, 3)
    return pygame.surfarray.make_surface(img_rgb)


def rotate_image_3d_perspective(
    surf,
    theta,
    axis_point,
    axis_dir,
    cam_dist,
    f,
    screen_center
):
    """
    surf를 임의 축 주변으로 3D 회전시킨 뒤 원근 변환된 새 Surface 반환.
    axis_point가 이미지 밖이어도 문제 없음.
    """
    w, h = surf.get_width(), surf.get_height()

    # 원본 이미지의 3D 꼭짓점(중심을 원점으로)
    corners_local = np.array([
        [-w/2, -h/2, 0],
        [ w/2, -h/2, 0],
        [ w/2,  h/2, 0],
        [-w/2,  h/2, 0],
    ], dtype=np.float32)

    # 3D 회전
    corners_rot = rodrigues_rotate(corners_local, axis_point, axis_dir, theta)

    # 2D 투영
    corners_2d = project_points(
        corners_rot,
        cam_dist=cam_dist,
        f=f,
        screen_center=screen_center
    ).astype(np.float32)

    # 원본 이미지 2D 좌표
    src = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    # 투영된 좌표의 bounding box를 새 surface 크기로
    min_xy = corners_2d.min(axis=0)
    max_xy = corners_2d.max(axis=0)
    out_w, out_h = (max_xy - min_xy).astype(int) + 2

    # 워핑 대상 좌표를 (0,0) 기준으로 이동
    dst = corners_2d - min_xy

    # homography
    M = cv2.getPerspectiveTransform(src, dst)

    # cv2 warp
    img_cv = surf_to_cv(surf)
    warped = cv2.warpPerspective(
        img_cv, M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0,0)
    )

    out_surf = cv_to_surf(warped)
    blit_pos = min_xy  # 화면에 붙일 위치(좌상단)

    return out_surf, blit_pos




def rotate_image(img, theta: float, cam_dist, f, screen_center,
         axis_point: np.array, axis_dir: np.array = np.array([0.0, 1.0, 0.0], dtype=np.float32)):

    cam_dist = 900.0
    f = 700.0
    screen_center = (400, 300)

    rotated_surf, pos = rotate_image_3d_perspective(
        img, theta,
        axis_point=axis_point,
        axis_dir=axis_dir,
        cam_dist=cam_dist,
        f=f,
        screen_center=screen_center
    )

    pos = (float(pos[0]), float(pos[1]))

    return rotated_surf, pos

