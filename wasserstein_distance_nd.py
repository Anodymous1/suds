import numpy as np
from numpy import asarray

from scipy import sparse
from scipy.spatial import distance_matrix

from scipy.optimize import milp, LinearConstraint



from scipy.stats._stats_py import _cdf_distance
from scipy.stats._stats_py import _validate_distribution


def wasserstein_distance_nd(u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute the Wasserstein-1 distance between two N-D discrete distributions.

    The Wasserstein distance, also called the Earth mover's distance or the
    optimal transport distance, is a similarity metric between two probability
    distributions [1]_. In the discrete case, the Wasserstein distance can be
    understood as the cost of an optimal transport plan to convert one
    distribution into the other. The cost is calculated as the product of the
    amount of probability mass being moved and the distance it is being moved.
    A brief and intuitive introduction can be found at [2]_.

    .. versionadded:: 1.13.0

    Parameters
    ----------
    u_values : 2d array_like
        A sample from a probability distribution or the support (set of all
        possible values) of a probability distribution. Each element along
        axis 0 is an observation or possible value, and axis 1 represents the
        dimensionality of the distribution; i.e., each row is a vector
        observation or possible value.

    v_values : 2d array_like
        A sample from or the support of a second distribution.

    u_weights, v_weights : 1d array_like, optional
        Weights or counts corresponding with the sample or probability masses
        corresponding with the support values. Sum of elements must be positive
        and finite. If unspecified, each value is assigned the same weight.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    Given two probability mass functions, :math:`u`
    and :math:`v`, the first Wasserstein distance between the distributions
    using the Euclidean norm is:

    .. math::

        l_1 (u, v) = \inf_{\pi \in \Gamma (u, v)} \int \| x-y \|_2 \mathrm{d} \pi (x, y)

    where :math:`\Gamma (u, v)` is the set of (probability) distributions on
    :math:`\mathbb{R}^n \times \mathbb{R}^n` whose marginals are :math:`u` and
    :math:`v` on the first and second factors respectively. For a given value
    :math:`x`, :math:`u(x)` gives the probability of :math:`u` at position
    :math:`x`, and the same for :math:`v(x)`.

    This is also called the optimal transport problem or the Monge problem.
    Let the finite point sets :math:`\{x_i\}` and :math:`\{y_j\}` denote
    the support set of probability mass function :math:`u` and :math:`v`
    respectively. The Monge problem can be expressed as follows,

    Let :math:`\Gamma` denote the transport plan, :math:`D` denote the
    distance matrix and,

    .. math::

        x = \text{vec}(\Gamma)          \\
        c = \text{vec}(D)               \\
        b = \begin{bmatrix}
                u\\
                v\\
            \end{bmatrix}

    The :math:`\text{vec}()` function denotes the Vectorization function
    that transforms a matrix into a column vector by vertically stacking
    the columns of the matrix.
    The transport plan :math:`\Gamma` is a matrix :math:`[\gamma_{ij}]` in
    which :math:`\gamma_{ij}` is a positive value representing the amount of
    probability mass transported from :math:`u(x_i)` to :math:`v(y_i)`.
    Summing over the rows of :math:`\Gamma` should give the source distribution
    :math:`u` : :math:`\sum_j \gamma_{ij} = u(x_i)` holds for all :math:`i`
    and summing over the columns of :math:`\Gamma` should give the target
    distribution :math:`v`: :math:`\sum_i \gamma_{ij} = v(y_j)` holds for all
    :math:`j`.
    The distance matrix :math:`D` is a matrix :math:`[d_{ij}]`, in which
    :math:`d_{ij} = d(x_i, y_j)`.

    Given :math:`\Gamma`, :math:`D`, :math:`b`, the Monge problem can be
    transformed into a linear programming problem by
    taking :math:`A x = b` as constraints and :math:`z = c^T x` as minimization
    target (sum of costs) , where matrix :math:`A` has the form

    .. math::

        \begin{array} {rrrr|rrrr|r|rrrr}
            1 & 1 & \dots & 1 & 0 & 0 & \dots & 0 & \dots & 0 & 0 & \dots &
                0 \cr
            0 & 0 & \dots & 0 & 1 & 1 & \dots & 1 & \dots & 0 & 0 &\dots &
                0 \cr
            \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots
                & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots  \cr
            0 & 0 & \dots & 0 & 0 & 0 & \dots & 0 & \dots & 1 & 1 & \dots &
                1 \cr \hline

            1 & 0 & \dots & 0 & 1 & 0 & \dots & \dots & \dots & 1 & 0 & \dots &
                0 \cr
            0 & 1 & \dots & 0 & 0 & 1 & \dots & \dots & \dots & 0 & 1 & \dots &
                0 \cr
            \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots &
                \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \cr
            0 & 0 & \dots & 1 & 0 & 0 & \dots & 1 & \dots & 0 & 0 & \dots & 1
        \end{array}

    By solving the dual form of the above linear programming problem (with
    solution :math:`y^*`), the Wasserstein distance :math:`l_1 (u, v)` can
    be computed as :math:`b^T y^*`.

    The above solution is inspired by Vincent Herrmann's blog [3]_ . For a
    more thorough explanation, see [4]_ .

    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] "Wasserstein metric",
           https://en.wikipedia.org/wiki/Wasserstein_metric
    .. [2] Lili Weng, "What is Wasserstein distance?", Lil'log,
           https://lilianweng.github.io/posts/2017-08-20-gan/#what-is-wasserstein-distance.
    .. [3] Hermann, Vincent. "Wasserstein GAN and the Kantorovich-Rubinstein
           Duality". https://vincentherrmann.github.io/blog/wasserstein/.
    .. [4] Peyré, Gabriel, and Marco Cuturi. "Computational optimal
           transport." Center for Research in Economics and Statistics
           Working Papers 2017-86 (2017).

    See Also
    --------
    wasserstein_distance: Compute the Wasserstein-1 distance between two
        1D discrete distributions.

    Examples
    --------
    Compute the Wasserstein distance between two three-dimensional samples,
    each with two observations.

    >>> from scipy.stats import wasserstein_distance_nd
    >>> wasserstein_distance_nd([[0, 2, 3], [1, 2, 5]], [[3, 2, 3], [4, 2, 5]])
    3.0

    Compute the Wasserstein distance between two two-dimensional distributions
    with three and two weighted observations, respectively.

    >>> wasserstein_distance_nd([[0, 2.75], [2, 209.3], [0, 0]],
    ...                      [[0.2, 0.322], [4.5, 25.1808]],
    ...                      [0.4, 5.2, 0.114], [0.8, 1.5])
    174.15840245217169
    """
    m, n = len(u_values), len(v_values)
    u_values = asarray(u_values)
    v_values = asarray(v_values)

    if u_values.ndim > 2 or v_values.ndim > 2:
        raise ValueError('Invalid input values. The inputs must have either '
                         'one or two dimensions.')
    # if dimensions are not equal throw error
    if u_values.ndim != v_values.ndim:
        raise ValueError('Invalid input values. Dimensions of inputs must be '
                         'equal.')
    # if data is 1D then call the cdf_distance function
    if u_values.ndim == 1 and v_values.ndim == 1:
        return _cdf_distance(1, u_values, v_values, u_weights, v_weights)

    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)
    # if number of columns is not equal throw error
    if u_values.shape[1] != v_values.shape[1]:
        raise ValueError('Invalid input values. If two-dimensional, '
                         '`u_values` and `v_values` must have the same '
                         'number of columns.')

    # if data contains np.inf then return inf or nan
    if np.any(np.isinf(u_values)) ^ np.any(np.isinf(v_values)):
        return np.inf
    elif np.any(np.isinf(u_values)) and np.any(np.isinf(v_values)):
        return np.nan

    # create constraints
    A_upper_part = sparse.block_diag((np.ones((1, n)), ) * m)
    A_lower_part = sparse.hstack((sparse.eye(n), ) * m)
    # sparse constraint matrix of size (m + n)*(m * n)
    A = sparse.vstack((A_upper_part, A_lower_part))
    A = sparse.coo_array(A)

    # get cost matrix
    D = distance_matrix(u_values, v_values, p=2)
    cost = D.ravel()

    # create the minimization target
    p_u = np.full(m, 1/m) if u_weights is None else u_weights/np.sum(u_weights)
    p_v = np.full(n, 1/n) if v_weights is None else v_weights/np.sum(v_weights)
    b = np.concatenate((p_u, p_v), axis=0)

    # solving LP
    constraints = LinearConstraint(A=A.T, ub=cost)
    opt_res = milp(c=-b, constraints=constraints, bounds=(-np.inf, np.inf))
    return -opt_res.fun
