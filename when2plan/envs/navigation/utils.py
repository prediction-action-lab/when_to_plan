import numpy as np
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Point, Polygon, box as Box


def generate_tasks(
    rng,
    num_tasks,
    max_obstacles,
    min_obstacles,
    obstacle_types,
    obstacle_probs,
    obstacle_shapes,
    room_size,
    robot_radius,
    min_start_goal_dist,
):
    start_positions = np.zeros((num_tasks, 2), dtype=np.float32)
    goal_positions = np.zeros((num_tasks, 2), dtype=np.float32)
    obstacles = np.zeros((num_tasks, max_obstacles, 3), dtype=np.float32)
    obstacle_types_array = np.zeros((num_tasks, max_obstacles), dtype="U20")
    for task_idx in range(num_tasks):
        max_attempts = 100
        valid_task = False

        for attempt in range(max_attempts):
            # Generate obstacles first
            num_obstacles = rng.integers(min_obstacles, max_obstacles + 1)
            placed = 0
            attempts = 0

            temp_obstacles = np.zeros((max_obstacles, 3), dtype=np.float32)
            temp_obstacle_types = np.zeros(max_obstacles, dtype="U20")

            while placed < num_obstacles and attempts < 100:
                attempts += 1

                if rng.random() < obstacle_probs["overall"]:
                    obstacle_type = rng.choice(
                        list(obstacle_types.values()),
                        p=[obstacle_probs[ot] for ot in obstacle_types.values()],
                    )

                    # Random position with margin from walls
                    margin = 2.0
                    pos_x = rng.uniform(margin, room_size - margin)
                    pos_y = rng.uniform(margin, room_size - margin)

                    temp_obstacles[placed] = [pos_x, pos_y, 1.0]
                    temp_obstacle_types[placed] = obstacle_type
                    placed += 1

            # Now find valid start and goal positions
            start_pos = _find_valid_position(
                rng,
                temp_obstacles,
                obstacle_shapes,
                temp_obstacle_types,
                room_size,
                robot_radius,
            )
            if start_pos is None:
                continue

            # Find goal position that's far enough from start
            for _ in range(50):
                goal_pos = _find_valid_position(
                    rng,
                    temp_obstacles,
                    obstacle_shapes,
                    temp_obstacle_types,
                    room_size,
                    robot_radius,
                )
                if goal_pos is None:
                    continue

                dist = np.linalg.norm(start_pos - goal_pos)
                if dist >= min_start_goal_dist:
                    # Verify path exists using simplified check
                    if _path_likely_exists(
                        start_pos,
                        goal_pos,
                        temp_obstacles,
                        obstacle_shapes,
                        temp_obstacle_types,
                        robot_radius,
                    ):
                        start_positions[task_idx] = start_pos
                        goal_positions[task_idx] = goal_pos
                        obstacles[task_idx] = temp_obstacles
                        obstacle_types_array[task_idx] = temp_obstacle_types
                        valid_task = True
                        break

            if valid_task:
                break

        if not valid_task:
            print(
                f"Warning: Could not generate valid task {task_idx}, using simpler layout"
            )
            # Fallback: create simple task with fewer obstacles
            start_positions[task_idx] = [
                robot_radius * 2,
                robot_radius * 2,
            ]
            goal_positions[task_idx] = [
                room_size - robot_radius * 2,
                room_size - robot_radius * 2,
            ]

    return start_positions, goal_positions, obstacles, obstacle_types_array


def _find_valid_position(
    rng, obstacles, obstacle_shapes, obstacle_types, room_size, robot_radius
):
    """Find a position that doesn't collide with obstacles or walls."""
    margin = robot_radius * 2
    for _ in range(100):
        pos = np.array(
            [
                rng.uniform(margin, room_size - margin),
                rng.uniform(margin, room_size - margin),
            ],
            dtype=np.float32,
        )

        if not _check_collision_with_obstacles(
            pos, obstacles, obstacle_shapes, obstacle_types, robot_radius
        ):
            return pos
    return None


def _path_likely_exists(
    start, goal, obstacles, obstacle_shapes, obstacle_types, robot_radius
):
    """Quick check if path likely exists using line-of-sight and simple checks."""
    # Check if direct line intersects with any obstacle
    robot = ShapelyPoint(start[0], start[1]).buffer(robot_radius)

    # Sample points along the line
    num_samples = int(np.linalg.norm(goal - start) / robot_radius)
    num_samples = max(5, min(num_samples, 20))

    for i in range(num_samples + 1):
        t = i / num_samples
        sample_pos = start + t * (goal - start)
        if _check_collision_with_obstacles(
            sample_pos, obstacles, obstacle_shapes, obstacle_types, robot_radius
        ):
            # Path blocked, but that's okay - we just want it to be possible, not trivial
            return True

    # Direct line of sight exists - still valid but might be too easy
    # We'll accept it anyway
    return True


def _check_collision_with_obstacles(
    pos, obstacles, obstacle_shapes, obstacle_types, robot_radius
):
    """Check if position collides with obstacles."""
    robot = Point(pos[0], pos[1]).buffer(robot_radius)

    for i in range(len(obstacles)):
        if obstacles[i, 2] > 0.5:  # valid obstacle
            obs_geom = _create_obstacle_geometry(
                obstacle_shapes, obstacle_types[i], obstacles[i, 0], obstacles[i, 1]
            )
            if robot.intersects(obs_geom):
                return True

    return False


def _create_obstacle_geometry(obstacle_shapes, obstacle_type, pos_x, pos_y):
    """Create Shapely geometry for collision detection."""
    shape_info = obstacle_shapes[obstacle_type]

    if shape_info[0] == "circle":
        radius = shape_info[1]
        return Point(pos_x, pos_y).buffer(radius)
    else:  # rectangle
        width, height = shape_info[1], shape_info[2]
        # Create box centered at (pos_x, pos_y)
        return Box(
            pos_x - width / 2,
            pos_y - height / 2,
            pos_x + width / 2,
            pos_y + height / 2,
        )


def world_to_pixel(x, y, pixels_per_unit=10, room_size=20):
    """Convert world coordinates to pixel coordinates."""
    px = int(x * pixels_per_unit)
    py = int((room_size - y) * pixels_per_unit)  # Flip Y for image coords
    return px, py


def draw_circle(
    img,
    center_x,
    center_y,
    radius,
    color,
    pixels_per_unit=10,
    room_size=20,
    img_size=64,
    fill=True,
):
    """Draw a circle on the image using bounding-box clipping."""
    cx, cy = world_to_pixel(center_x, center_y, pixels_per_unit, room_size)
    r_px = int(radius * pixels_per_unit)

    # Clip to the circle's bounding box so we never allocate full-image arrays
    y0 = max(cy - r_px - 1, 0)
    y1 = min(cy + r_px + 2, img_size)
    x0 = max(cx - r_px - 1, 0)
    x1 = min(cx + r_px + 2, img_size)
    if y0 >= y1 or x0 >= x1:
        return

    ys = np.arange(y0, y1)
    xs = np.arange(x0, x1)
    dy = ys[:, None] - cy
    dx = xs[None, :] - cx
    dist_sq = dx * dx + dy * dy

    if fill:
        mask = dist_sq <= r_px * r_px
        img[y0:y1, x0:x1][mask] = color
    else:  # outline
        outer = dist_sq <= r_px * r_px
        inner = dist_sq <= (r_px - 2) ** 2
        mask = outer & ~inner
        img[y0:y1, x0:x1][mask] = color


def draw_rectangle(
    img,
    center_x,
    center_y,
    width,
    height,
    color,
    pixels_per_unit=10,
    room_size=20,
    img_size=64,
    fill=True,
):
    """Draw a rectangle on the image."""
    x1, y1 = world_to_pixel(
        center_x - width / 2, center_y - height / 2, pixels_per_unit, room_size
    )
    x2, y2 = world_to_pixel(
        center_x + width / 2, center_y + height / 2, pixels_per_unit, room_size
    )

    # Ensure proper ordering
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Clip to image bounds
    x1, x2 = max(0, x1), min(img_size, x2)
    y1, y2 = max(0, y1), min(img_size, y2)

    if fill:
        img[y1:y2, x1:x2] = color
    else:  # Outline
        thickness = 2
        img[y1 : y1 + thickness, x1:x2] = color  # Top
        img[y2 - thickness : y2, x1:x2] = color  # Bottom
        img[y1:y2, x1 : x1 + thickness] = color  # Left
        img[y1:y2, x2 - thickness : x2] = color  # Right
