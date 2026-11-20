import numpy as np

def generate_vin_style_gridworld(rng, height, width, obstacle_probability):
    """
    Generates a gridworld with randomly placed obstacles, VIN-style.
    0 represents an obstacle, 1 represents a free path.

    Args:
        key: JAX random key.
        height: Height of the grid.
        width: Width of the grid.
        obstacle_probability: Probability that an internal cell is an obstacle.

    Returns:
        grid: A JAX array representing the gridworld.
    """
    # Start with all paths
    grid = np.ones((height, width), dtype=np.int32)

    # Create border walls
    grid[0, :] = 0  # Top border
    grid[-1, :] = 0  # Bottom border
    grid[:, 0] = 0  # Left border
    grid[:, -1] = 0  # Right border

    # Generate random obstacles for internal cells
    # The shape for internal cells is (height-2, width-2)
    if height > 2 and width > 2:  # Ensure there are internal cells
        # bernoulli gives True if it's an obstacle (with obstacle_probability)
        is_obstacle = rng.random((height - 2, width - 2)) < obstacle_probability
        # Convert boolean (True for obstacle) to int (0 for obstacle, 1 for path)
        internal_cells = 1 - is_obstacle.astype(np.int32)
        grid[1:-1, 1:-1] = internal_cells

    return grid


def generate_tasks(cfg, seed, map_size):
    """
    Generates a batch of gridworlds using NumPy.
    Assumes cfg contains: seed, n_tasks, map_size, obstacle_probability
    """
    # Use default_rng for a modern, good quality random number generator
    rng = np.random.default_rng(seed)
    obstacle_prob = cfg.env.obstacle_probability

    grids = []
    for i in range(cfg.env.num_tasks):
        # Pass the rng object directly, it manages its own state
        grid = generate_vin_style_gridworld(rng, map_size, map_size, obstacle_prob)
        grids.append(grid)

    return np.stack(grids)


# Example of the recursive division maze (can be kept or removed)
def generate_recursive_division_maze(rng, height, width):
    """
    Generates a maze using the recursive division algorithm, using NumPy.

    Args:
        rng: A NumPy random number generator (e.g., np.random.default_rng()).
        height: Height of the maze (must be odd)
        width: Width of the maze (must be odd)

    Returns:
        maze: A NumPy array representing the maze (0 = wall, 1 = path)
    """
    # Ensure odd dimensions
    height = height if height % 2 == 1 else height + 1
    width = width if width % 2 == 1 else width + 1

    # Initialize maze with all walls
    maze = np.zeros((height, width), dtype=np.int32)

    # Set entire area as a chamber (1 = path)
    maze[1:-1, 1:-1] = 1

    def divide_chamber(maze_state, x1, y1, x2, y2, current_rng):
        """Recursively divides a chamber into smaller chambers."""
        chamber_width = x2 - x1 + 1
        chamber_height = y2 - y1 + 1

        # Base case: chamber is too small to divide
        if chamber_width <= 1 or chamber_height <= 1:
            return maze_state

        # Choose a random horizontal or vertical division
        if chamber_width > chamber_height:
            horizontal = False  # Vertical division
        elif chamber_height > chamber_width:
            horizontal = True  # Horizontal division
        else:
            horizontal = current_rng.choice([True, False])  # Random choice

        new_maze_state = maze_state

        if horizontal:
            # Horizontal division
            num_possible_wall_rows = (chamber_height - 1) // 2
            if num_possible_wall_rows <= 0:
                return new_maze_state

            wall_y_offset = current_rng.integers(0, num_possible_wall_rows)
            wall_y = y1 + 2 * wall_y_offset + 1

            new_maze_state[wall_y, x1 : x2 + 1] = 0  # Add horizontal wall

            num_possible_passage_cols = chamber_width // 2
            if num_possible_passage_cols <= 0:
                return new_maze_state

            passage_x_offset = current_rng.integers(0, num_possible_passage_cols)
            passage_x = x1 + 2 * passage_x_offset
            passage_x = np.clip(passage_x, x1, x2)
            if passage_x % 2 == 0:
                passage_x = np.clip(passage_x + 1, x1, x2)

            new_maze_state[wall_y, passage_x] = 1

            new_maze_state = divide_chamber(
                new_maze_state, x1, y1, x2, wall_y - 1, current_rng
            )
            new_maze_state = divide_chamber(
                new_maze_state, x1, wall_y + 1, x2, y2, current_rng
            )
        else:
            # Vertical division
            num_possible_wall_cols = (chamber_width - 1) // 2
            if num_possible_wall_cols <= 0:
                return new_maze_state

            wall_x_offset = current_rng.integers(0, num_possible_wall_cols)
            wall_x = x1 + 2 * wall_x_offset + 1
            new_maze_state[y1 : y2 + 1, wall_x] = 0  # Add vertical wall

            num_possible_passage_rows = chamber_height // 2
            if num_possible_passage_rows <= 0:
                return new_maze_state

            passage_y_offset = current_rng.integers(0, num_possible_passage_rows)
            passage_y = y1 + 2 * passage_y_offset
            passage_y = np.clip(passage_y, y1, y2)
            if passage_y % 2 == 0:
                passage_y = np.clip(passage_y + 1, y1, y2)

            new_maze_state[passage_y, wall_x] = 1

            new_maze_state = divide_chamber(
                new_maze_state, x1, y1, wall_x - 1, y2, current_rng
            )
            new_maze_state = divide_chamber(
                new_maze_state, wall_x + 1, y1, x2, y2, current_rng
            )

        return new_maze_state

    # Start the recursive division
    # We use a single RNG instance and pass it down.
    # The in-place modifications to `maze` are fine with NumPy.
    maze = divide_chamber(maze, 1, 1, width - 2, height - 2, rng)

    return maze