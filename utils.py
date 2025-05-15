def read_paths(file_path):
    with open(file_path, 'r') as file:
        paths = [line.strip() for line in file.readlines()]
    return paths


def create_polygon_points(x, y):
    # Check if x and y are single integers
    if isinstance(x, int) and isinstance(y, int):
        polygon_points = [(x, y)]
    else:
        # Convert single integers to lists
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        # Create polygon points
        polygon_points = [(xi, yi) for xi, yi in zip(x, y)]

    return polygon_points