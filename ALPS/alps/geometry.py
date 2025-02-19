import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import math
import networkx as nx
from itertools import combinations
import alps.vector as vec

def connect_dots(points):
    """
    Connect the dots in a list of points to form a closed loop with the minimum path length.
    The algorithm starts with the convex hull of the points and then inserts the remaining points
    ... This algorithm is a simple heuristic and may not always find the optimal solution,
        especially for large N and complex point distributions.

    Parameters
    ----------
    points: np.ndarray, shape (nPoints, nDim)

    Returns
    -------
    ordered_points: np.ndarray, shape (nPoints, nDim)
    """
    def segments_intersect(p1, p2, q1, q2, tol=1e-6):
        """
        Check if two line segments (p1, p2) and (q1, q2) intersect in N-dimensional space.
        Parameters:
            p1, p2: Endpoints of the first segment (arrays of shape (N,))
            q1, q2: Endpoints of the second segment (arrays of shape (N,))
            tol: Tolerance for floating-point comparisons
        Returns:
            True if the segments intersect, False otherwise
        """
        # Vector directions of the segments
        d1 = p2 - p1
        d2 = q2 - q1

        # Check if the segments are parallel by checking if their cross product is zero
        # For N-dimensions, use determinant-based methods or general checks for collinearity
        def are_parallel(v1, v2):
            # Create a matrix with v1 and v2 and check rank
            mat = np.vstack([v1, v2])
            rank = np.linalg.matrix_rank(mat)
            return rank < 2  # If rank < 2, vectors are linearly dependent (parallel)

        if are_parallel(d1, d2):
            # Check if the segments lie on the same line or subspace
            if np.allclose(np.cross(d1, q1 - p1), 0, atol=tol):
                # Check for overlap in their parameterized representation
                t0 = np.dot(q1 - p1, d1) / np.dot(d1, d1)
                t1 = np.dot(q2 - p1, d1) / np.dot(d1, d1)
                if (0 <= t0 <= 1 or 0 <= t1 <= 1 or
                        (t0 < 0 and t1 > 1) or (t1 < 0 and t0 > 1)):
                    return True
            return False

        # For non-parallel segments, check if they intersect in the common subspace
        # Solve for intersection using parameterization of the segments:
        # p1 + t * d1 = q1 + s * d2 -> solve for t and s
        A = np.column_stack((d1, -d2))
        b = q1 - p1

        if np.linalg.matrix_rank(A) == A.shape[1]:  # Check if solution exists
            t_s = np.linalg.lstsq(A, b, rcond=None)[0]
            t, s = t_s

            # Check if the intersection point lies within the segments' bounds
            if 0 <= t <= 1 and 0 <= s <= 1:
                return True

        return False

    # Step 1: Compute the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    remaining_points = np.delete(points, hull.vertices, axis=0)

    # Step 2: Construct initial non-intersecting path from the convex hull
    ordered_points = list(hull_points)

    # Step 3: Incrementally insert remaining points while maintaining order
    # print(ordered_points)
    # remaining_points = [p for p in points if not any(np.array_equal(p, hp) for hp in ordered_points)]
    # print('Total points: ', len(points), 'Ordered points', len(ordered_points), 'Remaining points: ', len(remaining_points)
    # )

    for p in remaining_points:
        best_position = None
        min_new_length = float('inf')

        # Try inserting p between every pair of consecutive points in the loop
        for i in range(len(ordered_points)):
            new_loop = ordered_points[:i + 1] + [p] + ordered_points[i + 1:]

            # Calculate the new path length
            new_length = sum(np.linalg.norm(new_loop[j] - new_loop[j + 1]) for j in range(len(new_loop) - 1))
            if new_length < min_new_length:
                min_new_length = new_length
                best_position = i

        # Insert p at the best found position
        if best_position is not None:
            ordered_points = ordered_points[:best_position + 1] + [p] + ordered_points[best_position + 1:]

    # Step 4: Close the loop
    ordered_points.append(ordered_points[0])

    # Display the ordered points (Note: visualization for N > 3 isn't feasible without dimensionality reduction)
    ordered_points = np.asarray(ordered_points) # nPoints x nDim

    return ordered_points

def connect_phis(phis, close=True):
    """
    Connect the volume fractions in phis to form a closed loop in the (nComp-1)-dimensional space.

    Parameters
    ----------
    phis: np.ndarray, shape (nComp, nPhases) ; nPhases > 1
        ... Volume fractions at the phase boundaries

    Returns
    -------
    ordered_phis: np.ndarray, shape (nComp, nPhases)
    """
    # Step 1: Connect phis[:-1, :] (shape: nComp-1 x nPhases) to form a closed loop
    ## NEVER PASS A FULL phis ARRAY TO connect_dots FUNCTION
    ## BECAUSE PHIS live on nComp-1 dimensional space, one must connect phis in the nComp-1 dimensional space
    ## We will recover the last component later.
    if phis.shape[1] < 3:
        return phis
    else:
        try:
            tmp = connect_dots(phis[:-1, :].T).T # tmp.shape = nComp-1 x nPts + 1
        except:
            print("Error in connect_dots, phi.shape", phis.shape)
            return phis
        # Step 2: Recover the last component via phi_n = 1 - sum(phi_i for i in range(n-1))
        ordered_phis = np.empty_like(phis)
        ordered_phis[:-1, :] = tmp[:, :-1]
        ordered_phis[-1, :] = 1 - np.nansum(tmp[:, :-1], axis=0)
        if close:
            ordered_phis = np.hstack((ordered_phis, ordered_phis[:, 0][:, np.newaxis]))
        return ordered_phis

def get_facets(phis):
    """
    Get all possible (nComp-1)-dimensional polytopes (facets) from the nComp-dimensional polytopes.

    Parameters
    ----------
    phis: list of np.ndarray, shape (nComp, nPoints), Dimension of a polytope is nPoints

    Returns
    -------
    polytopes: list of np.ndarray of shape (nComp, nPoints = nComp-1),
        ... e.g. Ternary phase diagram (nComp = 3)
            ... [edge 1 - shape: (nComp, nPoints), edge 2 (nComp, nPoints), edge 3 (nComp, nPoints)]
        ... Length of the list is `num_comp`
    centers: list of np.ndarray of shape (nPoints, )
        ... e.g. [Centroid on edge 1 - shape: (nComp, ), Centroid on edge 2 (nComp, ), Centroid on edge 3 (nComp, )]
        ... Length of the list is `num_comp`
    normals: list of np.ndarray of shape (nComp-1, nPoints)
        ... e.g. [Normal to edge 1 - shape: (nComp, ), Normal to edge 2 (nComp, ), Normal to edge 3 (nComp, )]
        ... Length of the list is `num_comp`
    """
    phis = np.asarray(phis)
    phi_center = np.mean(phis, axis=1)
    num_comp, npts = phis.shape
    # Choose the all possible n-1 components
    indices_list = list(combinations(range(num_comp), num_comp-1))
    facets = [phis[:, indices] for indices in indices_list] # [(nComp-1, nPoints), ...]
    centers = [np.mean(phis[:, indices], axis=1) for indices in indices_list] # [(nComp-1, ), ...]
    normals = [vec.get_normal(polytope[:-1, :]) for polytope in facets] # [(nComp, nPoints) -> (nComp, ), ...]

    directions = [p_center[:-1] - phi_center[:-1] for p_center in centers]
    # Check if the direction of the normal vector is pointing away from the center
    for i, (direction, normal) in enumerate(zip(directions, normals)):
        if np.dot(direction, normal) < 0:
            normals[i] = -normals[i]

    # Complete the last component of the normal vector
    ## The last component of the normal vector is the negative sum of the other components
    ## to satisfy the requirement:
    ## ... \sum_i^N \Delta phi_i = 0
    ## where \Delta phi = phi_next - phi_prev = const * normal
    ## This requirement comes from (1) \sum_i^N phi_i = 1 for phi = phi_next and phi = phi_prev.
    last_components = - np.sum(normals, axis=1)
    normals = np.concatenate((np.asarray(normals), last_components), axis=1)
    normals = [vec.normalize(normal) for normal in normals]
    return facets, centers, normals

def get_k_faces(phis, k, debug=False):
    """
    (UNDER CONSTRUCTION)
    Get all k-faces from the N-dimensional polytopes.
    ... M is the number of components in the phase diagram.
    ... Volume fraction, phi, lives in the (M-1)-dimensional space.

    Therefore, any object in the phase diagram is a polytope in the (N-1)-dimensional space.
    Terminology: A k-face is a k-polytope embedded in N-polytope.
    ... N-polytope is a polytope in the N-dimensional space.
    ... N-faces are the N-polytopes themselves!
    ... (N-1)-faces are called Facets.
    ... (N-2)-faces are called Ridges.
    ... (N-3)-faces are called Peaks.
    ... 0-faces are called Vertices.

    References to Liquid-Liquid Phase Separation:
    ... IMPORTANT: Volume fraction, phi, lives in the (nComp-1)-dimensional space.
    ...... (nComp = number of components)
    ... The lost degree of freedom is due to the constraint: \sum_i^N \phi_i = 1

    Example 1: Ternary phase diagram (nComp = 3)
    ... Phase space: 2-Polytope (faces)
    ... Dimension of Phase Space: N = nComp - 1 = 2
    ... Faces:
    ...... 2-face (face): Triphase (Complete separation)
    ...... 1-face (edge): Biphase (Two-phase separation)
    ...... 0-face (vertex): Monophase (Single phase / Mixed)

    Example 2: Quarternary phase diagram (nComp = 4)
    ... Phase space: 3-Polytope (faces)
    ... Dimension of Phase Space: N = nComp - 1 = 3
    ... Faces:
    ...... 3-face (cells): Tetraphase (Complete separation)
    ...... 2-face (face): Triphase (Three-phase separation)
    ...... 1-face (edge): Biphase (Two-phase separation)
    ...... 0-face (vertex): Monophase (Single phase / Mixed)

    # Prefixes: Mono-, bi-, tri-, tetra-, penta-, hexa-, hepta-, octa-, nona-, deca-, ...

    Parameters
    ----------
    phis: list of np.ndarray with shape (nComp, N). [\vec{phi^1}, \vec{phi^2}, ..., \vec{phi^N}]
    ... \vec{phi^i} is the volume fraction of the i-th phase.
    ... In David Zwicker's code, phis = phases.fractions.T (shape: ((nComp, nPhases, nComp)) (nPhases, nComp)
    ... Dimension of a polytope is N <= Nc - 1.
    ... e.g. Tie-lines (N = 1), phis.shape = (nComp, N+1) - Two points in the phase diagram
    ... e.g. Triangle (N = 2), phis.shape = (nComp, N+1) - Three points in the phase diagram
    ... e.g. Tetrahedron (N = 3), phis.shape = (nComp, N+1) - Four points in the phase diagram
    k: int, dimension of the polytopes to extract

    Returns
    -------
    polytopes: list of np.ndarray of shape (nComp, nPoints = nComp-k),
        ... e.g. Ternary phase diagram (nComp = 3)
            ... [edge 1 - shape: (nComp, nPoints), edge 2 (nComp, nPoints), edge 3 (nComp, nPoints)]
        ... Length of the list is equal to the number of k-faces `nKfaces = (N + 1) choose  (k + 1)`
    centers: list of np.ndarray of shape (nPoints, )
        ... e.g. [Centroid on edge 1 - shape: (nComp, ), Centroid on edge 2 (nComp, ), Centroid on edge 3 (nComp, )]
        ... Length of the list is equal to the number of k-faces `nKfaces = (N + 1) choose  (k + 1)`
    normals: list of np.ndarray of shape (nComp-1, nPoints)
        ... e.g. [Normals to edge 1 - shape: (nComp, nComp - k), Normals to edge 2 (nComp, nComp - k), Normals to edge 3 (nComp, nComp - k)]
        ... Length of the list is equal to the number of k-faces `nKfaces = (N + 1) choose  (k + 1)`
    """

    nComp, nPhas = M, P =phis.shape
    N = nComp - 1 # Dimension of the embedded vector space: \phi \in R^{M}
    # kmax = nPhas - 1 # Dimension of the faces e.g. Tie-line (k = 1), Tri-phase (k=2)

    if k > N:
        raise ValueError(f"k must be less than or equal to {phis.shape[1] - 1}")

    num_k_face = math.comb(N + 1, k + 1) # Number of k-faces in the N-polytope

    # Step 1: Get all possible k-faces from the N-dimensional polytopes
    indices_list = list(combinations(range(nPhas), k+1))
    # Shape: [(nComp, k+1), ..., (nComp, k+1)]
    kFaces = [phis[:, indices] for indices in indices_list]

    # Step 2: Compute the centroids of the k-faces
    # Shape: [(nComp, ), ..., (nComp, )]
    centroids = [np.mean(phis[:, indices], axis=1) for indices in indices_list]

    # Step 3: Construct a basis for each k-face
    # Construct a basis for each k-face
    ## basis = bases[0] = [e1, e2, ..., ek, |  e_(k+1)..., e_nComp], shape: (nComp, nComp)
    ## This basis is orthonormalized, and spans the k-face using the first k basis vectors.
    ## The last nComp - k basis vectors are the normal vectors to the k-face.
    ##  E.g. Ternary phase diagram (nComp = 3), k = 2 (triangle)
    ##      basis = [e1, e2 | n1], shape: (3, 3)
    ##  ... Ternary phase diagram can be spanned by two basis vectors e1, e2!
    ##  ... This is due to the constraint \sum_i^N \phi_i = 1.
    ##  ... The normal vector n1 is the normal to the triangle. Always stored in the last.
    ##  ... This is an unnecessary basis vector to span physically allowed space but useful.
    bases = [vec.construct_basis_from_polytope(kFace) for kFace in kFaces]

    # Step 4: Compute the normal vectors to the k-faces
    # Step 4-1: Extract normal vectors from the orthonormal basis

    ## Now we have the basis vectors for each k-face.
    ## To effectively scan the phase space, we need to know the normal vectors to the k-faces.
    ## For each k-face, the first k basis vectors span the k-face.
    ## The rest, nComp - k vectors, are the normal vectors to the k-face.
    ## The very last vector is normal to the physically allowed vector space.
    ## For example, in the ternary phase diagram, the normal vector is the normal to the triangle,
    ## defined by \sum_i^N \phi_i = 1.

    # 'normals', shape: [(nComp, nComp - k), ..., (nComp, nComp - k)]
    normals = [basis[:, k:] for basis in bases]
    if debug:
        print('debug: N, k ', N, k)
        print('debug basis[0].shape: ',bases[0].shape)
        print('debug normals[0].shape: ',normals[0].shape)


    # Step 4-2: A trick to effectively move along the normal vectors
    # In the phase diagram, we will move as
    # ... \phi_next = \phi_prev + step * normal
    # The last component of \phi is determined by the constraint \sum_i^N \phi_i = 1.
    # So there is no consequence in adjusting the last component of the normal vector.
    # To ensure we move by `step` in the phase diagram, we adjust the last component of
    # the normal vector as follows:
    # ... normal[-1] = - \sum_{i=1}^{nComp - 1} normal_i

    # Sketch of derivation:
    # ... | \phi_next[:-1] - \phi_prev[:-1] |^2 = step^2
    # ... \sum \phi_next = 1, \sum \phi_prev = 1
    def adjust_normal(normal):
        """
        Adjust the last component of the normal vector to ensure the step size in the phase diagram.
        """
        normal[-1, :] = - np.sum(normal[:-1, :], axis=0)
        return normal

    # After the adjustment, normalize the normal
    # Exception handling: If k = N = nComp - 1, the normal vector cannot ve defined! Return an empty list.
    normals = [vec.normalize(adjust_normal(normal)) for normal in normals if N != k]
    if debug:
        print('debug normals (nComp, k): ', normals)


    return kFaces, centroids, normals

