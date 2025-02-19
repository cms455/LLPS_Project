import numpy as np
from sympy import Matrix
import itertools

# def get_normal_legacy(points):
#     """
#     Returns a normal vector to the hyperplane defined by the points. Applicable to N-dimensional space.
#
#     ... points: (nComp, nPoints), points = (\vec{p1}, \vec{p2}, ..., \vec{pN})
#         where \vec{p1} = (x1, x2, ..., xN) is a point in N-dimensional space.
#     ... augmented matrrix
#                 A = (e1, e2, ..., eN, 1)
#                     (  \vec{p1}^T   , 1)
#                     (  \vec{p2}^T   , 1)
#                     (     ...       , 1)
#                     (  \vec{pN}^T   , 1)
#     ... normal vector = det(A)
#
#     Parameters:
#     points (numpy.ndarray): A numpy array of shape (N, NumPoints = N), where N is the number of dimensions.
#     ... points must be linearly independent.
#
#     Returns:
#     numpy.ndarray: A normal vector to the hyperplane.
#     """
#     # Check that the input is valid
#     if points.shape[1] != points.shape[0]:
#         raise ValueError("The number of points must be equal to the number of dimensions.")
#
#     # Create an augmented matrix with basis vectors and the points
#     augmented_matrix = np.zeros((points.shape[0] + 1, points.shape[0] + 1), dtype=object)
#
#     # Fill the first row with basis vector symbols
#     for i in range(points.shape[0]):
#         augmented_matrix[0, i] = f'e{i + 1}'
#     augmented_matrix[:, -1] = 1
#     # Fill the remaining rows with the points and an appended 1
#     augmented_matrix[1:, :-1] = points.T
#
#     # Calculate the determinant symbolically
#     det = Matrix(augmented_matrix).det()
#
#     # Convert determinant to a numpy array
#     normal = np.array([float(det.coeff(f'e{i + 1}')) for i in range(points.shape[0])])
#     normal /= np.linalg.norm(normal)
#     return normal

def get_normal(polytope, return_basis=False):
    """
    Get a set of the normal vectors to the polytope.
    ... N-polytope is a generalizaiton of a polygon.
    ...... 0-polytope: Point
    ...... 1-polytope: Line
    ...... 2-polytope: Polygon
    ...... 3-polytope: Polyhedron, and so on.

    ... Shape of the polytope: (nComp, nPoints)

    ... A given polytope could be embedded in a higher-dimensional space. nComp > (nPoints - 1).
    ...... For example, a line in 3D space is a 1-polytope embedded in 3D space. nComp = 3, nPoints = 2.
    ...... For a line, one needs two points.
    ...... In such case, two normals can be defined to the line.
    ...... With the tangent vector, the three vectors form a basis in 3D space.
    ...... This function returns the normal vectors to the polytope.


    Parameters
    ----------
    polytopes: numpy arrays.
        ... Each numpy array has a shape of (nComp, nPoints).
        where nComp is the number of components and nPoints is the number of points.
    return_basis: bool, default: False
        ... If True, return the basis vectors of the polytope.

    Returns
    -------
    normals: numpy array, shape (nComp, nComp - nPoints + 1)
        ... Normal vectors to the polytope.
    basis: numpy array, shape (nComp, nComp), optional
    """
    nPoints = polytope.shape[1]
    k = nPoints - 1 # degree of the polytope # k=0: point, k=1: line, k=2: plane, k=3: volume, ...

    # Orthonormalize the basis
    ## Basis vectors are stored as columns- basis[:, i] is the i-th basis vector
    basis = construct_basis_from_polytope(polytope)
    normals = basis[:, k:] # normal vectors, (nComp, nComp-k)
    if return_basis:
        return normals, basis
    else:
        return normals

def construct_basis_from_polytope(polytope):
    """
    Construct a basis from points of a given polytope.
    ... Polytope is an N-polytope, a generalization of a polygon. Shape: (nComp, nPoints)
    ... The basis is orthonormalized.
    ... A given polytope could be embedded in a higher-dimensional space. nComp > (nPoints - 1).

    Parameters
    ----------
    polytope: numpy array, shape (nComp, nPoints)
    ... nComp is the number of components and nPoints is the number of points.

    Returns
    -------
    basis: numpy array, shape (nComp, nComp)
    ... orthonormal basis
    ... basis[:, i] is the i-th basis vector.
    ... basis[:, :nPoints-1] are the vectors that span the polytope.
    ... basis[:, nPoints-1:] are the normal vectors to the polytope.
    """

    nPoints = polytope.shape[1]
    k = nPoints - 1  # degree of the polytope # k=0: point, k=1: line, k=2: plane, k=3: volume, ...
    embedding_dim = polytope.shape[0]  # dimension of the embedding space

    vectors = []
    vectors_candidates = [v2 - v1 for v1, v2 in itertools.combinations(polytope.T, 2)]
    for v in vectors_candidates:
        if len(vectors) == k:
            break
        if is_linearly_independent(v, vectors):
            vectors.append(normalize(v))

    # Append random, linearly independent vectors to make the basis
    while True:
        if len(vectors) == embedding_dim:
            break
        v = create_independent_vector(np.random.rand(embedding_dim))

        if is_linearly_independent(v, vectors):
            vectors.append(normalize(v))

    # Orthonormalize the basis
    ## Basis vectors are stored as columns- basis[:, i] is the i-th basis vector
    basis = gram_schmidt(vectors)  # orthonormal basis, ndarray, (nComp, nComp)
    return basis

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def is_linearly_independent(v, list_of_vectors, tol=0.01):
    """Check if vector v is linearly independent from the list of vectors"""
    for u in list_of_vectors:
        if np.abs(1 - np.dot(normalize(v), normalize(u))) < tol:
            return False
    return True

def gram_schmidt(initial_vectors, return_ndarray=True):
    """
    Perform the Gram-Schmidt process to generate an orthogonal basis in the vector space spanned by initial_vectors.

    Parameters:
    - initial_vectors: list of numpy arrays, the initial set of vectors.

    Returns:
    - basis: list of orthonormal numpy arrays forming the basis.
    """
    # Determine the dimension of the vector space
    dim = len(initial_vectors[0])
    # Check if the initial vectors are linearly independent
    independent_vectors = []
    for v in initial_vectors:
        if is_linearly_independent(v, independent_vectors):
            independent_vectors.append(v)

    # Append additional independent vectors if needed
    while len(independent_vectors) < dim:
        new_vector = np.random.randn(dim)
        if is_linearly_independent(new_vector, independent_vectors):
            independent_vectors.append(new_vector)

    # Perform Gram-Schmidt procedure
    basis = []
    for v in independent_vectors:
        for b in basis:
            v = v - np.dot(v, b) * b
        if np.linalg.norm(v) > 1e-10:
            basis.append(normalize(v))
    if return_ndarray:
        # Basis vectors are stored as columns- basis[:, i] is the i-th basis vector
        basis = np.asarray(basis).T
    return basis


def create_independent_vector(v, tol=1e-1):
    """
    Generate an N-dimensional vector that is independent of the given vector v.

    Parameters:
    - v: numpy array, the given vector.

    Returns:
    - independent_vector: numpy array, a vector that is linearly independent of v.
    """

    N = v.shape[0]  # Dimensionality of the vector space

    while True:
        # Generate a random N-dimensional vector
        independent_vector = np.random.rand(N)
        # Check if it is independent by ensuring it's not collinear (dot product is not 1 or -1)
        if np.abs(1 - np.dot(normalize(v), normalize(independent_vector))) < tol:
            return independent_vector
