import jax
import jax.numpy as np

def create_mesh_functions(points, cells):
    @jax.jit
    def get_updated_elements(sol):
        updated_coordinates = points + sol.reshape(points.shape)
        return jax.vmap(lambda cell: updated_coordinates[cell])(cells)
    
    @jax.jit
    def get_element_edges(element_nodes):
        nodes = element_nodes
        next_nodes = np.roll(element_nodes, -1, axis=0)
        edges = np.stack([nodes, next_nodes], axis=1)
    
        return edges
    
    def compute_boundary_edges(cells):
        # 1. generate all the edges of the cells
        node_edges = jax.vmap(get_element_edges)(cells) 
        flat_edges = node_edges.reshape(-1, 2)
       # 2. sort the edges and create unique ID
        sorted_edges = np.sort(flat_edges, axis=1)  
        edge_ids = sorted_edges[:,0] * (np.max(cells) + 1) + sorted_edges[:,1]
       # 3. sort the edge_ids, the same edges will be adjacent
        sorted_idx = np.argsort(edge_ids)
        sorted_ids = edge_ids[sorted_idx]
       # 4. find the edges that appear only once, by comparing the adjacent elements
        is_start = np.concatenate([
            np.array([True]), 
            sorted_ids[1:] != sorted_ids[:-1]
        ])
        is_end = np.concatenate([
            sorted_ids[1:] != sorted_ids[:-1],
            np.array([True])
        ])
        is_unique = is_start & is_end
    # 5. return the boundary edges in the original order
        boundary_idx = sorted_idx[is_unique]
        boundary_edges = flat_edges[boundary_idx]
        return boundary_edges

    def find_boundary_edges(contact_elements):
        """
        find the boundary edges
        the boundary edges, each row contains the two nodes of an edge
        """
        contact_cells = cells[contact_elements]
        boundary_edges = compute_boundary_edges(contact_cells)
        return boundary_edges

    def get_node_coords(sol, boundary_edges):
        """get the current coordinates of the boundary edges
            (starts, ends): the current coordinates of the boundary edges
        """
        start_nodes = boundary_edges[:, 0]
        end_nodes = boundary_edges[:, 1]
        sol = sol.reshape(-1, 2)
        starts = np.take(points, start_nodes, axis=0) + sol[start_nodes]
        ends = np.take(points, end_nodes, axis=0) + sol[end_nodes]
        return starts, ends

        
    return {
        'get_updated_elements': get_updated_elements,
        'compute_boundary_edges': compute_boundary_edges,
        'get_node_coords': get_node_coords,
        'find_boundary_edges': find_boundary_edges
    }


def is_point_in_element(point, element_vertices):
    """use the vectorized cross product to judge if the point is in the quadrilateral cell
    """

    edges = np.roll(element_vertices, -1, axis=0) - element_vertices  # (4, 2)

    to_points = point - element_vertices  # (4, 2)

    crosses = edges[:, 0] * to_points[:, 1] - edges[:, 1] * to_points[:, 0]  # (4,)
    all_positive = np.all(crosses >= 0)
    all_negative = np.all(crosses <= 0)

    return np.logical_or(all_positive, all_negative)

@jax.jit
def is_point_in_elastic_body(point, updated_elements):
    """ if the point is in the elastic body"""
    return jax.vmap(lambda element: is_point_in_element(
        point, element))(updated_elements).any()

@jax.jit
def is_point_in_rigid_body(point, rigid_body_params):
    """if the point is in the rigid body"""
    edges = np.roll(rigid_body_params, -1, axis=0) - rigid_body_params  
    to_points = point - rigid_body_params  

    crosses = edges[:, 0] * to_points[:, 1] - edges[:, 1] * to_points[:, 0]  
    
    all_positive = np.all(crosses >= 0)
    all_negative = np.all(crosses <= 0)
    return np.logical_or(all_positive, all_negative)

@jax.jit
def compute_segment_sdf(point, segment_starts, segment_ends):
    """
    compute the distance from the point to the segments
    """
    ba = segment_ends - segment_starts  
    pa = point - segment_starts        
    h = np.clip(
        np.sum(pa * ba, axis=1) / np.sum(ba * ba, axis=1),
        0.0, 1.0
    )  
    distances = np.linalg.norm(
        pa - h[:, None] * ba,
        axis=1
    )  
    return distances  


def compute_elastic_boundary_sdf(point, starts, ends, contact_cells):
    """compute the SDF of the point to the elastic body"""

    distances = compute_segment_sdf(point, starts, ends)  

    min_distance = np.min(distances)

    is_inside = is_point_in_elastic_body(point, contact_cells)
    return np.where(is_inside, -min_distance, min_distance)

def compute_rigid_boundary_sdf(point, rigid_body_params):
    """compute the SDF of the point to the rigid body
    """

    starts = rigid_body_params  
    ends = np.roll(rigid_body_params, -1, axis=0)  

    distances = compute_segment_sdf(point, starts, ends)  
    min_distance = np.min(distances)
    
    is_inside = is_point_in_rigid_body(point, rigid_body_params)
    
    return np.where(is_inside, -min_distance, min_distance)


def compute_contact_boundary_sdf1(point, starts, ends, contact_cells, rigid_body_params):
    """
    compute the SDF of the point to the contact boundary for elastic-rigid contact
    """
    # 1. compute the SDF of the point to the elastic body
    sdf_elastic = compute_elastic_boundary_sdf(point, starts, ends, contact_cells)
    # 2. compute the SDF of the point to the rigid body
    sdf_rigid = compute_rigid_boundary_sdf(point, rigid_body_params)
    # 3. return the maximum of the two SDFs
    return np.maximum(sdf_elastic, sdf_rigid)


def compute_contact_boundary_sdf2(point, starts1, ends1, contact_elements1, starts2, ends2, contact_elements2):
    """compute the SDF of the point to the contact boundary for elastic-elastic contact
    """
    sdf_elastic1 = compute_elastic_boundary_sdf(point, starts1, ends1, contact_elements1)
    sdf_elastic2 = compute_elastic_boundary_sdf(point, starts2, ends2, contact_elements2)
    return np.maximum(sdf_elastic1, sdf_elastic2)
