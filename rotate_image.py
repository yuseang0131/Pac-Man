import math
import numpy as np
import cv2
import pygame


# ---------------------------
# 3D 회전 (Rodrigues 공식)
# ---------------------------
def rodrigues_rotate(points, axis_point, axis_dir, theta):
    """
    points: (N, 3) float32  - 회전시킬 점들
    axis_point: (3,) float32 - 회전축이 지나는 한 점
    axis_dir: (3,) float32   - 회전축 방향 벡터 (정규화 안 되어 있어도 됨)
    theta: float             - 회전각 (rad)
    """
    points = np.asarray(points, dtype=np.float32)
    axis_point = np.asarray(axis_point, dtype=np.float32)
    axis_dir = np.asarray(axis_dir, dtype=np.float32)

    # 축 방향 정규화
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 축 기준으로 평행이동
    p = points - axis_point  # (N,3)

    k = axis_dir  # (3,)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Rodrigues 공식
    # p_rot = p*cosθ + (k×p)*sinθ + k(k·p)(1−cosθ)
    cross = np.cross(k, p)               # (N,3)
    dot = np.dot(p, k)[:, None]          # (N,1)

    p_rot = p * cos_t + cross * sin_t + k * dot * (1.0 - cos_t)

    # 원래 위치로 되돌리기
    return p_rot + axis_point


# ---------------------------
# 3D -> 2D 투영 (단순 카메라 모델)
# ---------------------------
def project_points(points, cam_dist=1000.0):
    """
    points: (N, 3)
    cam_dist: 카메라가 화면 뒤쪽 z축 방향으로 떨어진 거리 (>0)

    기준 좌표계:
      - 원래 이미지가 z=0 평면 위에 있음.
      - 카메라는 (0,0,-cam_dist)에 있고, +z 방향을 바라본다고 가정.
      - 스크린(모니터)은 z=0 평면.
    z=0인 점은 그대로 (x,y)로 매핑됨.
    z가 +로 갈수록 더 멀어지므로 작게 보임.
    """
    points = np.asarray(points, dtype=np.float32)
    z = points[:, 2]

    # scale = cam_dist / (cam_dist + z)
    eps = 1e-6
    scale = cam_dist / (cam_dist + z + eps)

    x_2d = points[:, 0] * scale
    y_2d = points[:, 1] * scale

    return np.stack([x_2d, y_2d], axis=1)  # (N,2)


# ---------------------------
# pygame <-> OpenCV 변환
# ---------------------------
def pygame_surface_to_cv2_rgba(surface: pygame.Surface) -> np.ndarray:
    """
    pygame.Surface -> OpenCV RGBA (H, W, 4) uint8
    """
    surf = surface.convert_alpha()
    w, h = surf.get_size()
    data = pygame.image.tostring(surf, "RGBA", False)
    arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))
    # OpenCV는 기본이 BGR/BGRA지만 투명도까지 살리려면 일단 RGBA 유지
    return arr


def cv2_rgba_to_pygame_surface(arr: np.ndarray) -> pygame.Surface:
    """
    OpenCV RGBA (H, W, 4) -> pygame.Surface
    """
    h, w, _ = arr.shape
    surf = pygame.image.frombuffer(arr.tobytes(), (w, h), "RGBA")
    return surf


# ---------------------------
# 핵심 함수: 3D 회전 후 warpPerspective
# ---------------------------
def rotate_image_3d(
    surface: pygame.Surface,
    img_center_2d,          # (cx, cy) 화면 상에서 이미지 중심 좌표
    axis_point_3d,          # (ax, ay, az) 회전축이 지나는 한 점
    axis_dir_3d,            # (dx, dy, dz) 회전축 방향 벡터
    theta_rad,              # 회전각 (rad)
    base_z=0.0,             # ✅ 이 이미지가 처음에 위치할 z값
    cam_dist=1000.0,
):
    """
    pygame 이미지(surface)를 주어진 3D 회전축 기준으로 회전시키고,
    OpenCV 퍼스펙티브 워핑으로 새 surface와 blit할 위치를 반환.

    base_z: 이미지 전체가 처음에 위치하는 z 좌표 (기본 0.0)
    """
    w, h = surface.get_size()
    cx, cy = img_center_2d

    # 1. 회전 전 이미지 4 꼭짓점의 3D 좌표
    #    👉 z를 전부 base_z로 둔다
    corners_3d = np.array(
        [
            [cx - w / 2.0, cy - h / 2.0, base_z],  # top-left
            [cx + w / 2.0, cy - h / 2.0, base_z],  # top-right
            [cx + w / 2.0, cy + h / 2.0, base_z],  # bottom-right
            [cx - w / 2.0, cy + h / 2.0, base_z],  # bottom-left
        ],
        dtype=np.float32,
    )

    # 2. 회전축 기준 3D 회전
    rotated_corners_3d = rodrigues_rotate(
        corners_3d,
        axis_point=np.array(axis_point_3d, dtype=np.float32),
        axis_dir=np.array(axis_dir_3d, dtype=np.float32),
        theta=theta_rad,
    )

    # 3. 3D -> 2D 투영
    dst_pts_2d = project_points(rotated_corners_3d, cam_dist=cam_dist)  # (4,2)

    # 4. bounding box 계산
    min_xy = np.floor(dst_pts_2d.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(dst_pts_2d.max(axis=0)).astype(np.int32)
    dst_w, dst_h = (max_xy - min_xy).tolist()

    if dst_w <= 0 or dst_h <= 0:
        # 너무 극단적인 경우 보호
        return surface, (int(cx - w / 2), int(cy - h / 2))

    dst_pts_local = dst_pts_2d - min_xy.astype(np.float32)

    src_pts = np.array(
        [
            [0.0, 0.0],
            [w - 1.0, 0.0],
            [w - 1.0, h - 1.0],
            [0.0, h - 1.0],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts_local.astype(np.float32))

    img_rgba = pygame_surface_to_cv2_rgba(surface)

    warped_rgba = cv2.warpPerspective(
        img_rgba,
        M,
        (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    rotated_surface = cv2_rgba_to_pygame_surface(warped_rgba)
    dst_pos = (int(min_xy[0]), int(min_xy[1]))

    return rotated_surface, dst_pos

# ---------------------------
# 간단한 데모 (원하면 사용)
# ---------------------------

# ---------------------------
# 여러 이미지를 화면 중심 축으로 3D 공전시키는 데모
# ---------------------------
def main_demo():
    pygame.init()
    screen_w, screen_h = 800, 600
    screen = pygame.display.set_mode((screen_w, screen_h))
    clock = pygame.time.Clock()

    base_img = pygame.image.load("data/imgs/Wall/end00.png").convert_alpha()
    base_img = pygame.transform.scale(base_img, (120, 120))

    screen_center = (screen_w / 2.0, screen_h / 2.0, 0.0)

    # 공전 축 방향 (y축 기준으로 회전)
    axis_dir = (0.0, 1.0, 0.0)

    cam_dist = 1200.0

    num_images = 5
    radius = 200.0
    images = []
    for i in range(num_images):
        angle = 2 * math.pi * i / num_images
        cx = screen_center[0] + radius * math.cos(angle)
        cy = screen_center[1] + radius * math.sin(angle)

        # ✅ 이미지별 초기 z값 (예시로 -150 ~ +150 사이로 분포)
        base_z = (i - (num_images - 1) / 2.0) * 75.0

        images.append(
            {
                "surface": base_img.copy(),
                "center": (cx, cy),   # (x, y)
                "z": base_z,          # ✅ 초기 z
                "phase": angle,
            }
        )

    running = True
    theta = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        theta += 0.01

        screen.fill((30, 30, 30))

        pygame.draw.circle(
            screen,
            (80, 255, 80),
            (int(screen_center[0]), int(screen_center[1])),
            5,
        )

        for info in images:
            img = info["surface"]
            img_center = info["center"]
            base_z = info["z"]

            local_theta = theta  # 필요하면 + info["phase"] 등으로 위상 조정 가능

            rotated_img, pos = rotate_image_3d(
                img,
                img_center_2d=img_center,
                axis_point_3d=screen_center,
                axis_dir_3d=axis_dir,
                theta_rad=local_theta,
                base_z=base_z,       # ✅ 여기서 각 이미지의 초기 z 전달
                cam_dist=cam_dist,
            )

            screen.blit(rotated_img, pos)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main_demo()
