import open3d
import numpy as np


# Store pressure values of the sensor array
SENSOR_ARRAY = None


def hex2RGB(color):
    """
    :param color: in the hex format
    :return: [R, G, B] values in range (0, 1)
    """
    color = color.lstrip('#')
    R, G, B = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
    rgb = [R / 255, G / 255, B / 255]
    return rgb


def open_simulation(stimuli):
    """
    Creates Visualization objects and adds all the objects to the display
    :param stimuli: External object with fixed radius chosen by the user
    """

    # Create plain surface + sensors
    sensor_mesh, sensor_cloud = create_mesh(width=10, height=10)

    vis = open3d.visualization.Visualizer()
    vis.create_window(
        window_name='main_window',
        width=10 * 100, height=7 * 100,
    )
    # Load rendering options from a .json file
    vis.get_render_option().load_from_json("render_option.json")

    # Add objects to the visualization
    vis.add_geometry(sensor_mesh)
    vis.add_geometry(sensor_cloud)
    # vis.add_geometry(stimuli)

    # TODO position camera so that whole mesh is visible
    recenter_camera(sensor_mesh, vis)

    # TODO
    vis.register_animation_callback(on_animation_callback)

    # Run the visualizer
    vis.run()
    vis.destroy_window()


def on_animation_callback(vis):
    vis.poll_events()
    vis.update_renderer()


def recenter_camera(obj, vis):
    """
    # TODO
    :param obj: To be position in the middle of the frame
    :param vis: Visualization instance
    """
    center = obj.get_center()


def create_mesh(width, height):
    """
    :param width: number of taxels horizontally
    :param height: number of taxels vertically
    :return: Triangular mesh and point cloud corresponding to sensor positions
    """
    global SENSOR_ARRAY

    # FIXME unit for the visualization
    DISTANCE_BETWEEN_SENSORS = 2.5

    # Triangulation method
    vertices, triangles = [], []
    step = DISTANCE_BETWEEN_SENSORS

    for i in range(height - 1):
        for j in range(width - 1):
            a = [i * step, j * step, 0]
            b = [(i + 1) * step, j * step, 0]
            c = [(i + 1) * step, (j + 1) * step, 0]
            d = [i * step, (j + 1) * step, 0]

            vertices.append(a)
            vertices.append(b)
            vertices.append(c)
            vertices.append(d)

            index = len(vertices) - 1
            triangles.append([index - 3, index - 2, index - 1])
            triangles.append([index - 3, index - 1, index])

    np_vertices, np_triangles = np.array(vertices), np.array(triangles).astype(np.int32)

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(np_vertices)
    mesh.triangles = open3d.utility.Vector3iVector(np_triangles)

    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(np_vertices)

    SENSOR_ARRAY = np.zeros(np_vertices.shape)

    # Create shadows
    mesh.compute_vertex_normals()

    mesh.paint_uniform_color(hex2RGB('#5da2ff'))
    cloud.paint_uniform_color(hex2RGB('#ff3dd6'))
    return mesh, cloud

# TODO
'''
1) recenter_camera()

check if relative sizes are ok
Add side panel with instruction



open simulation
move mesh with mouse 
Create taxel array object with unique ID and make it to correspond to cloudaaray
output real time pressure values from all the sensors, save to xsl file


'''
