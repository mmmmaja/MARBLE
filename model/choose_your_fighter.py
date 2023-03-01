import open3d
from simulation import open_simulation, hex2RGB


def create_sphere():
    mesh = open3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    mesh.paint_uniform_color(hex2RGB('#FF8360'))  # Red color for the sphere
    return mesh


def create_cylinder():
    mesh = open3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=1.0)
    mesh.paint_uniform_color(hex2RGB('#E8E288'))  # Green color for the cylinder
    return mesh

'#7DCE82'
'#3CDBD3'
'#00FFF5'

# Define the possible shapes and associated create functions
shapes = [
    create_sphere(),
    create_cylinder()
]

# Initialize the selected shape index and radius
selected_shape = 0


# Create a function to update the visualization
def update_visualization(vis):
    global selected_shape

    # Remove the old geometry
    vis.clear_geometries()

    # Create the new geometry and add it to the scene
    vis.add_geometry(shapes[selected_shape])

    # Update the title to display the selected shape and radius
    vis.update_renderer()


# Register the callback for the right arrow key
def next_shape_callback(vis):
    global selected_shape
    selected_shape = (selected_shape + 1) % len(shapes)
    update_visualization(vis)


# Register the callback for the right arrow key
def previous_shape_callback(vis):
    global selected_shape
    selected_shape = (selected_shape - 1) % len(shapes)
    update_visualization(vis)


def next_page_callback(vis):
    global selected_shape

    _radius = float(input("Enter the radius of the stimuli: "))
    vis.destroy_window()
    geometry = shapes[selected_shape]
    geometry.scale(_radius, center=geometry.get_center())
    open_simulation(geometry)


def initialize():
    # Create the visualization window
    _vis = open3d.visualization.VisualizerWithKeyCallback()
    _vis.create_window(
        window_name='choose_your_fighter',
        width=10 * 100, height=7 * 100,
    )
    _vis.register_key_callback(ord("D"), next_shape_callback)
    _vis.register_key_callback(ord("A"), previous_shape_callback)
    _vis.register_key_callback(ord("S"), next_page_callback)

    # Initialize the visualization
    _vis.get_render_option().load_from_json("render_option.json")
    update_visualization(_vis)

    # Run the visualization loop
    _vis.run()

    # Close the visualization window
    _vis.destroy_window()


def preset(index=0, radius=1):
    geometry = shapes[index]
    geometry.scale(radius, center=geometry.get_center())
    open_simulation(geometry)
