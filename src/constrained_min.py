import numpy as np
import math


def solve_optimization_problem(objective_function, inequality_constraints, equality_constraints_matrix,
                               equality_constraints_rhs, initial_point):
    path = []
    values = []
    scale = 1

    current_obj_val, current_gradient, current_hessian = calculate_objective_value(objective_function,
                                                                                   inequality_constraints,
                                                                                   initial_point, scale)

    num_inequality_constraints = len(inequality_constraints)
    current_point = initial_point.copy()

    path.append(current_point)
    values.append(objective_function(current_point)[0])

    while num_inequality_constraints / scale > 1e-8:
        for _ in range(10):
            direction = determine_optimal_direction(current_hessian, equality_constraints_matrix, current_gradient)
            step_size = perform_backtracking_line_search(objective_function, current_point, current_obj_val,
                                                         current_gradient, direction, inequality_constraints, scale)
            next_point = current_point + direction * step_size

            next_obj_val, next_gradient, next_hessian = calculate_objective_value(objective_function,
                                                                                  inequality_constraints, next_point,
                                                                                  scale)

            lambda_value = np.sqrt(np.dot(direction, np.dot(next_hessian, direction.T)))
            if 0.5 * (lambda_value ** 2) < 1e-8:
                break

            current_point = next_point
            current_obj_val = next_obj_val
            current_gradient = next_gradient
            current_hessian = next_hessian

        path.append(current_point.copy())
        values.append(objective_function(current_point.copy())[0])
        scale *= 10

    return current_point, objective_function(current_point.copy())[0], {'path': path, 'values': values}


def calculate_objective_value(objective_function, inequality_constraints, point, scale):
    obj_val, gradient, hessian = objective_function(point)
    log_obj_val, log_gradient, log_hessian = apply_logarithmic_barrier(inequality_constraints, point)
    updated_obj_val = scale * obj_val + log_obj_val
    updated_gradient = scale * gradient + log_gradient
    updated_hessian = scale * hessian + log_hessian
    return updated_obj_val, updated_gradient, updated_hessian


def apply_logarithmic_barrier(inequality_constraints, point):
    dimensions = point.shape[0]
    log_obj_val = 0
    log_gradient = np.zeros((dimensions,))
    log_hessian = np.zeros((dimensions, dimensions))

    for constraint in inequality_constraints:
        f_val, g_val, h_val = constraint(point)
        log_obj_val += math.log(-f_val)
        log_gradient += (1.0 / -f_val) * g_val

        gradient = g_val / f_val
        gradient_dim = gradient.shape[0]
        gradient_tile = np.tile(gradient.reshape(gradient_dim, -1), (1, gradient_dim)) * np.tile(
            gradient.reshape(gradient_dim, -1).T, (gradient_dim, 1))
        log_hessian += (h_val * f_val - gradient_tile) / f_val ** 2

    return -log_obj_val, log_gradient, -log_hessian


def determine_optimal_direction_with_equality(hessian, A, gradient):
    left_matrix = np.block([
        [hessian, A.T],
        [A, 0],
    ])
    right_vector = np.block([[-gradient, 0]])
    ans = np.linalg.solve(left_matrix, right_vector.T).T[0]
    return ans[0:A.shape[1]]


def determine_optimal_direction_without_equality(hessian, gradient):
    return np.linalg.solve(hessian, -gradient)


def determine_optimal_direction(hessian, A, gradient):
    if A is not None:
        return determine_optimal_direction_with_equality(hessian, A, gradient)
    return determine_optimal_direction_without_equality(hessian, gradient)


def perform_backtracking_line_search(objective_function, point, obj_val, gradient, direction, inequality_constraints,
                                     scale, alpha=0.01, beta=0.5, max_iter=10):
    step_size = 1
    current_obj_val, _, _ = objective_function(point + step_size * direction)

    iteration_count = 0
    while iteration_count < max_iter and current_obj_val > obj_val + alpha * step_size * gradient.dot(direction):
        step_size *= beta
        current_obj_val, _, _ = objective_function(point + step_size * direction)
        iteration_count += 1

    return step_size
