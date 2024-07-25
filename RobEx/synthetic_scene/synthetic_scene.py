import trimesh
import numpy as np


def bottom_point(mesh):
    mesh_dump = mesh.dump().sum()
    mesh_vertices = mesh_dump.vertices
    vertices_z = mesh_vertices[:, 1]
    bottom = np.min(vertices_z)

    return bottom


class ObjectMesh(object):
    def __init__(self):
        super(ObjectMesh, self).__init__()


if __name__ == "__main__":
    save = False
    y_axis = [0, 1, 0]
    x_axis = [1, 0, 0]

    scene = trimesh.Scene()
    room_file = (
        ""
    )
    room_mesh = trimesh.load(room_file, process=False)
    bottom_room = bottom_point(room_mesh)
    scene.add_geometry(room_mesh)

    mesh_objs = []

    chair2 = ObjectMesh()
    chair2.class_id = "03001627"
    chair2.instance_id = "336e92c7c570250251c4deb11af7079e"
    chair2.scale = 1.0
    chair2.angle = np.pi
    chair2.translate = np.array([0.6, 0.0, -1.0])
    mesh_objs.append(chair2)

    sofa1 = ObjectMesh()
    sofa1.class_id = "04256520"
    sofa1.instance_id = "9f6d960c57515fb491264d3b5d25d83f"
    sofa1.scale = 2.0
    sofa1.angle = -np.pi / 2
    sofa1.translate = np.array([-2.0, 0.0, 0.3])
    mesh_objs.append(sofa1)

    table1 = ObjectMesh()
    table1.class_id = "04379243"
    table1.instance_id = "bb00ad069df73df5d1f943907d4f35fc"
    table1.scale = 1.8
    table1.angle = 0.
    table1.translate = np.array([0., 0.0, 0.3])
    mesh_objs.append(table1)

    trashcan1 = ObjectMesh()
    trashcan1.class_id = "02747177"
    trashcan1.instance_id = "f7c95d62feb4fb03c0d62fb4fabc5d06"
    trashcan1.scale = 0.8
    trashcan1.angle = np.pi / 2
    trashcan1.translate = np.array([1.5, 0.0, -0.1])
    mesh_objs.append(trashcan1)

    for id, mesh_obj in enumerate(mesh_objs):
        mesh_obj.id = id
        file = (
            ""
            + mesh_obj.class_id
            + "/"
            + mesh_obj.instance_id
            + "/models/model_normalized.obj"
        )
        mesh = trimesh.load(file, process=False)
        angle = np.pi

        transform = trimesh.transformations.rotation_matrix(
            mesh_obj.angle, y_axis)
        mesh.apply_scale(mesh_obj.scale)
        mesh.apply_transform(transform)
        bottom_mesh = bottom_point(mesh)
        distance = bottom_room - bottom_mesh
        translation = mesh_obj.translate + np.array([0, distance, 0])
        mesh.apply_translation(translation)
        geom_name = (
            "obj_geom_" + str(id) + "." + mesh_obj.class_id +
            "." + mesh_obj.instance_id
        )
        node_name = (
            "obj_node_" + str(id) + "." + mesh_obj.class_id +
            "." + mesh_obj.instance_id
        )
        scene.add_geometry(mesh, geom_name=geom_name, node_name=node_name)

    angle = np.pi / 2
    scene_T = trimesh.transformations.rotation_matrix(angle, x_axis)
    scene.apply_transform(scene_T)

    if save:
        export = scene.export("scene.obj")
    scene.show()
