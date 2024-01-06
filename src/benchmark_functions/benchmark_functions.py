import numpy as np

def ackley_function(x: np.ndarray) -> float:
    """
    Ackley Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
    term2 = -np.exp(np.mean(np.cos(c * x)))
    return term1 + term2 + a + np.exp(1)

def eggholder_function(x: np.ndarray) -> float:
    """
    Eggholder Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    term1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + 0.5 * x[0] + 47)))
    term2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
    return term1 + term2

def griewank_function(x: np.ndarray) -> float:
    """
    Griewank Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    term1 = np.sum(x**2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + term1 - term2

def himmelblau_function(x: np.ndarray) -> float:
    """
    Himmelblau's Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    term1 = (x[0]**2 + x[1] - 11)**2
    term2 = (x[0] + x[1]**2 - 7)**2
    return term1 + term2

def levy_function(x: np.ndarray) -> float:
    """
    Levy Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0]))**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1))**2))
    term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
    return term1 + term2 + term3

def michalewicz_function(x: np.ndarray, m: int = 10) -> float:
    """
    Michalewicz Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.
    - m (int): Parameter controlling the shape of the function.

    Returns:
    - float: Objective function value at the given position.
    """
    return -np.sum(np.sin(x) * np.sin((np.arange(1, len(x) + 1) * x**2) / np.pi)**(2 * m))

def rastrigin_function(x: np.ndarray) -> float:
    """
    Rastrigin Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def schwefel_function(x: np.ndarray) -> float:
    """
    Schwefel Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def sphere_function(x: np.ndarray) -> float:
    """
    Sphere Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    return np.sum(x**2)
