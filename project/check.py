import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union

# Sample data setup
data = {
    'vis_navigation_plot': {
        'goal_position': np.array([8, 8]),  # example point
        'poi_position': np.array([[1, 1], [2, 2], [3, 3]]),  # example path points
        'poi_orientation': 0.5,  # example orientation in radians
        'base_positions': np.array([[0, 0], [1, 1], [2, 2]]),  # example base positions
        'path': np.array([[0, 0], [2, 2], [3, 3]]),  # example path
        'tree': np.array([[0, 0], [1, 1], [2, 2]]),  # example tree nodes
        'obstacles': [Polygon([(4, 4), (5, 5), (5, 4), (4, 4)])],  # example obstacles
        'lookahead_point': np.array([6, 6]),  # example lookahead point
        'node_polygon': Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)]),  # example node shape
        'node_anchor': np.array([0, 0])  # example anchor point for the node
    }
}

# Function to plot obstacles
def plot_obstacles(obstacles):
    for obstacle in obstacles:
        plt.plot(*obstacle.exterior.xy, color='red', label='Obstacle')

# Function to plot goal position
def plot_goal(goal_position):
    plt.scatter(*goal_position, c='green', s=50, label='Goal Position', zorder=10)

# Function to plot path points
def plot_path(path):
    plt.scatter([node[0] for node in path], [node[1] for node in path], c='blue', s=20, alpha=0.5, label='Path')

# Function to plot the node polygon after transformation
def plot_transformed_polygon(node_polygon, position, orientation, node_anchor):
    transformed_polygon = rotate(
        translate(node_polygon, xoff=position[0] - node_anchor[0], yoff=position[1] - node_anchor[1]),
        angle=orientation, use_radians=True
    )
    plt.plot(*transformed_polygon.exterior.xy, color='blue', label='Transformed Node Polygon')

# Function to plot the trajectory with an optional lookahead point
def plot_trajectory(poi_position, lookahead_point):
    plt.plot(poi_position[:, 0], poi_position[:, 1], color='orange', linestyle='--', linewidth=2, label='Trajectory')
    plt.scatter(*lookahead_point, color='magenta', s=50, label='Lookahead Point', zorder=10)

# Function to plot base positions
def plot_base_positions(base_positions):
    plt.scatter(base_positions[:, 0], base_positions[:, 1], color='orange', s=30, edgecolors='red', zorder=5, label='Base Positions')

# Function to plot swept hulls along the path
def plot_swept_hulls(path, node_polygon, node_anchor):
    hulls = []
    for i in range(len(path) - 1):
        start, end = path[i], path[i + 1]
        start_direction = np.arctan2(end[1] - start[1], end[0] - start[0])
        end_direction = start_direction  # Assume straight path for simplicity

        # Generate swept polygons
        swept_polygons = [
            rotate(translate(node_polygon, xoff=start[0] - node_anchor[0], yoff=start[1] - node_anchor[1]),
                   angle=angle, use_radians=True)
            for angle in np.linspace(start_direction, end_direction, 10)
        ]

        # Merge polygons for convex hull
        hull = MultiPolygon(swept_polygons).convex_hull.union(unary_union(swept_polygons))
        hulls.append(hull)

    unified_hull = unary_union(hulls)
    plt.plot(*unified_hull.exterior.xy, color='green', label='Swept Hull')

# Main plot function to display all components
def plot_environment(data):
    plt.figure(figsize=(10, 10))
    plt.clf()

    navigation_data = data['vis_navigation_plot']
    
    # Plot each component using modular functions
    plot_obstacles(navigation_data['obstacles'])
    plot_goal(navigation_data['goal_position'])
    plot_path(navigation_data['path'])
    plot_swept_hulls(navigation_data['path'], navigation_data['node_polygon'], navigation_data['node_anchor'])
    plot_transformed_polygon(navigation_data['node_polygon'], navigation_data['poi_position'][-1],
                             navigation_data['poi_orientation'], navigation_data['node_anchor'])
    plot_trajectory(navigation_data['poi_position'], navigation_data['lookahead_point'])
    plot_base_positions(navigation_data['base_positions'])

    # Final plot adjustments
    plt.title('Informed RRT* Environment')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.legend()
    plt.xlim(navigation_data['poi_position'][-1, 0] - 10, navigation_data['poi_position'][-1, 0] + 10)
    plt.ylim(navigation_data['poi_position'][-1, 1] - 10, navigation_data['poi_position'][-1, 1] + 10)

    plt.show()

# Call the main plotting function with sample data
plot_environment(data)
