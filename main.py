#
#
# def f(x, w):
#     return np.sin(x[:, 0] * w[0] + w[1])
#
#
# # def grad(f, x, w, delta=1e-5):
# #     size_w = len(w)
# #     grads = []
# #     for i in range(size_w):
# #         t = np.zeros(size_w)
# #         t[i] += delta
# #         wl = w + t
# #         wr = w - t
# #         ans = (f(x, wl) - f(x, wr)) / (2 * delta)
# #         grads.append(ans)
# #     return grads
# #
# #
# # def Jacobian(f, x, w):
# #     jacob = np.zeros((len(X), len(w)))
# #     for i in range(len(jacob)):
# #         jacob[i] = grad(f, x, w)
# #     return jacob
# def Jacobian(f, x, w):
#     eps = 1e-5
#     grads = []
#     for i in range(len(w)):
#         t = np.zeros_like(w).astype(float)
#         t[i] = t[i] + eps
#         grad = (f(x, w + t) - f(x, w - t)) / (2 * eps)
#         grads.append(grad)
#     return np.column_stack(grads)
#
#
# def Gauss_Newton(f, x, y, in_w, max_iter, eps=1e-5):
#     w = in_w
#     for i in range(max_iter):
#         last = w
#         J = Jacobian(f, x, w)
#         dy = y - f(x, w)
#         w += np.linalg.inv(J.T @ J) @ J.T @ dy
#         if np.linalg.norm(last - w) < eps:
#             break
#     return w, i
#
#
# x1 = np.linspace(-5, 5, 50)
# x2 = np.linspace(-5, 5, 50)
# X1, x2 = np.meshgrid(x1, x2)
# X = x1
# w = np.array([3, 3]).astype(float)
# y = f(X, w).astype(float) + np.random.normal(0, 1, size=len(X))
# bs = Gauss_Newton(f, X, y, w, 100)
# print(bs)
# from scipy.optimize import least_squares
#
#
# def func(w, X, y):
#     return f(X, w) - y
#
#
# res = least_squares(func, w, args=(X, y))
# print(res.x)
# import matplotlib.pyplot as plt
# import matplotlib as cm
# plt.plot(X, y, '')
# plt.plot(X, f(X, bs[0]), label='Real', )
# plt.show()
from collections import deque

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import line_search


def func(w, x):
    return (w[0] * x) ** 2 + np.cos(x) * w[1] ** 2


# def grad(w, x):
#     return w[1] + 2 * w[2] * x + w[]

def generate_points(n, f, rg=1):
    X = rg * np.random.uniform(0, 1, n)
    y = []
    X_err = X
    for x in X_err:
        y.append(f([5, 12], x))
    return X, np.asarray(y)


# хз но если применять jacob для подсчета градиента, плохо работает на сложных функциях(sin, log...)
def jacob(w, x, eps=1e-7):
    n, m = len(x), len(w)
    Jac = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            delta_p = w.copy()
            delta_p[j] -= eps
            fr = func(delta_p, x[i])
            delta_p[j] += 2 * eps
            fl = func(delta_p, x[i])
            Jac[i, j] = (fl - fr) / (2 * eps)
    return Jac


def test_jacob(w, x):
    jac = np.zeros((len(X), len(w)))
    for i in range(len(X)):
        jac[i] = np.array([2 * w[0] * x[i] ** 2, 2 * w[1] * np.cos(x[i])])
    return jac


def Gauss_Newton(f, x, y, init_w, max_iter=100, eps=1e-3):
    w = init_w
    points = [np.asarray(w)]
    for i in range(max_iter):
        J = test_jacob(w, x)
        dy = y - f(w, x)
        step = np.dot(np.linalg.pinv(J), dy)
        w += step
        if np.linalg.norm(step) < eps:
            break
        # w = w + np.linalg.inv(J.T @ J) @ J.T @ dy
        points.append(w)
    return w, np.asarray(points), i


X, y = generate_points(100, func)


# ans, points, i = Gauss_Newton(func, X, y, [1, 1])
# #
# # # Создаем график
# # plt.scatter(X, y, label='Исходные данные')
# # x_range = np.linspace(0, 1, 100)
# # y_range = func(ans, x_range)
# # plt.plot(x_range, y_range, 'r', label='Регрессия')
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
# print(ans, points, i)


#########
# DOGLEG#
#########


def dogleg_method(H_inv, J, H, trust_radius):
    deltaB = -np.dot(H_inv, J)
    norm_deltaB = np.linalg.norm(deltaB)

    if norm_deltaB <= trust_radius:
        return deltaB

    deltaU = - (np.dot(J, J) / np.dot(J, np.dot(H, J))) * J
    norm_deltaU = np.linalg.norm(deltaU)

    if norm_deltaU >= trust_radius:
        return trust_radius * deltaU / norm_deltaU

    diff_B_U = deltaB - deltaU
    diff_square_B_U = np.dot(diff_B_U, diff_B_U)
    dot_U_diff_B_U = np.dot(deltaU, diff_B_U)

    fact = dot_U_diff_B_U ** 2 - diff_square_B_U * (np.dot(deltaU, deltaU) - trust_radius ** 2)
    tau = (-dot_U_diff_B_U + math.sqrt(fact)) / diff_square_B_U

    return deltaU + tau * diff_B_U


def func(w, x):
    return (w[0] * x) ** 2 + np.cos(w[1] * x)


def func_calc(w, x):
    ans = 0
    for i in range(len(x)):
        ans += func(w, x[i])
    return ans


def grad(w, x):
    jac = np.zeros(len(w))
    for i in range(len(x)):
        jac[0] += 2 * w[0] * x[i] ** 2
        jac[1] += np.cos(w[1] * x[i])
    return jac


def hess(w, x):
    hessian = np.zeros((len(w), len(w)))
    n = len(x)
    for i in range(n):
        hessian[0][0] += 2 * x[i] ** 2
        hessian[0][1] += 0
        hessian[1][0] += 0
        hessian[1][1] += 4 * w[1] * math.cos(x[i])
    return hessian


def trust_region_dogleg(X, w_start, max_iters=100):
    eta = 0.2
    eps = 1e-3
    start_trust_radius = 1
    max_trust_radius = 10

    w = w_start
    trust_radius = start_trust_radius

    iters = 0
    jacobian_calc = 0
    hessian_calc = 0
    points = []
    while iters < max_iters:
        J = grad(w, X)
        jacobian_calc += 1

        H = hess(w, X)
        hessian_calc += 1

        H_inv = np.linalg.inv(H)
        delta = dogleg_method(H_inv, J, H, trust_radius)

        actual_reduction = func_calc(w, X) - func_calc(w + delta, X)

        predicted_reduction = -(np.dot(J, delta) + 0.5 * np.dot(delta, np.dot(H, delta)))
        if predicted_reduction == 0.0:
            ratio = 1e11
        else:
            ratio = actual_reduction / predicted_reduction

        delta_norm = np.linalg.norm(delta)
        if ratio < 0.2:
            trust_radius = 0.2 * delta_norm
        else:
            if ratio > 0.8 and delta_norm == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius

        if ratio > eta:
            w = w + delta
        else:
            w = w
            points.append(w)

        if np.linalg.norm(J) < eps:
            break

        if iters >= max_iters:
            break

        iters += 1
    return w


# ans = trust_region_dogleg(X, [1, 1])
# print(ans)


#######
# BFGS#
#######


def bfgs(start_x, func, grad, max_iter=10, ls=0.001, eps=1e-9):
    xk = start_x
    iter = 0
    I = np.eye(len(xk))
    Ik = I.copy()
    while iter < max_iter:
        grad_f = grad(xk)
        p = -np.dot(Ik, grad_f)
        ls = line_search(func, grad, xk, p)
        alpha = ls[0]
        xi = xk + alpha * p
        sk = xi - xk
        xk = xi

        new_grad_f = grad(xk)
        y = new_grad_f - grad_f
        grad_f = new_grad_f
        if (np.linalg.norm(grad_f) < eps):
            break
        rho = 1 / (y @ sk)
        a1 = I - rho * sk @ y.T
        a2 = I - rho * y @ sk.T
        Ik = (a1 @ (Ik @ a2.T)) + (rho * sk @ sk.T)
        # A1 = I - rho * sk[:, np.newaxis] * y[np.newaxis, :]
        # A2 = I - rho * y[:, np.newaxis] * sk[np.newaxis, :]
        # Ik = np.dot(A1, np.dot(Ik, A2)) + (rho * sk[:, np.newaxis] *
        #                                    sk[np.newaxis, :])
        iter += 1

    return xk, iter


def f(x):
    return 0.5 * x[0] ** 2 + np.sin(x[1]) * 2 * np.cos(x[0] + x[1] * 2)
    # x[0] ** 2 * np.log(x[1] * 12.5) + x[1] ** 2 * math.sin(x[0])


# x[0]**2 - x[0]*x[1] + x[1]**2 + 9*x[0] - 6*x[1] + 20
# def f1(x):
#     return np.array([2 * x[0] - x[1] + 9, -x[0] + 2*x[1] - 6])

def f1(x, delta=1e-8):
    n = len(x)
    ans = np.zeros(n, dtype=float)
    params = x.copy()

    for i in range(n):
        params[i] += delta
        fl = f(params)
        params[i] -= 2 * delta
        fr = f(params)
        params[i] += delta
        ans[i] = np.divide(fl - fr, 2 * delta)

    return ans


result, iter = bfgs([-0.3, -1.4], f, f1)

print('Result of BFGS method:')
print('Final Result (best point): %s' % (result))
print(iter)
print(f(result))

#######
# L-BFGS#
#######

def lbfgs(func, grad_func, x0, max_iter=10, m=10, eps=1e-7):
    result = [x0]
    xk = x0
    I = 1
    y_list, s_list, rho_list = [], [], []

    alpha = np.zeros(m)

    for i in range(max_iter):
        g = grad_func(xk)

        if i > m:
            for i in reversed(range(len(y_list))):
                alpha[i] = rho_list[i] * np.dot(s_list[i], g)
                g = g - alpha[i] * y_list[i]

        r = I * g

        if i > m:
            for i in range(len(y_list)):
                beta = rho_list[i] * np.dot(y[i], r)
                r = r + s_list[i] * (alpha[i] - beta)

        pk = -r
        a, _, _, _, _, _ = line_search(func, grad_func, xk, pk)

        if a is None:
            a = 1e-2
        new_xk = xk + a * pk

        if np.linalg.norm(g) < eps:
            break

        s_list.append(new_xk - xk)
        y_list.append(grad_func(new_xk) - g)
        rho_list.append(1 / (np.dot(y_list[-1],s_list[-1]) + eps))
        if len(s_list) > m:
            s_list.pop(0)
            y_list.pop(0)
            rho_list.pop(0)

        I = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])

        xk = new_xk
        result.append(xk)

    return result


result = lbfgs(f, f1, [-0.3, -1.4])
print(result)
print(f(result[-1]))
