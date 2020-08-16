import numpy as np
import quadpy


def test_quadpy_c1():
    if not hasattr(quadpy, 'c1'):
        return
    function_list = [
        (lambda x: np.array([1,2]).reshape(-1,1,1)*x),
        # (lambda x: np.array([0,2]).reshape(-1,1,1)*x), #fail
    ]
    for hf0 in function_list:
        _ = quadpy.c1.integrate_adaptive(hf0, [0,1], eps_abs=1e-5, eps_rel=1e-5)
    # quadpy.c1.integrate_adaptive(lambda x:x, [-1,1], eps_abs=1e-5, eps_rel=1e-5) #fail


def test_quadpy_line_segment():
    if not hasattr(quadpy, 'line_segment'):
        return
    function_list = [
        (lambda x: np.array([1,2]).reshape(-1,1,1)*x),
        # (lambda x: np.array([0,2]).reshape(-1,1,1)*x), #fail
    ]
    for hf0 in function_list:
        _ = quadpy.line_segment.integrate_adaptive(hf0, [0,1], eps=1e-5)
    _ = quadpy.line_segment.integrate_adaptive(lambda x:x, [-1,1], eps=1e-5)
