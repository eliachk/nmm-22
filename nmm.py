import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from time import time

l1 = 1
l2 = 1
b1 = 1
c = 3

# tol = 1e-9
# x = 0 if x < tol else x
# y = 0 if y < tol else y


def f(t, state):
    x, y = state
    dxdt = x * (-l1 + b1 * (y ** (2 / 3)) * (1 - x / c) / (1 + x))
    dydt = l2 * y - b2 * x * (y ** (2 / 3)) / (1 + x)
    return [dxdt, dydt]


def jacf(t, state):
    x, y = state
    dxx = -l1 + (b1 * y ** (2 / 3) * (c - x * (x + 2))) / c / ((x + 1) ** 2)
    dxy = 2 * b1 * x * (1-x/c) / 3 / (x + 1) / (y ** (1 / 3))
    dyx = -b2 * y ** (2 / 3) / ((x+1)**2)
    dyy = l2 - 2 * b2 * x / 3 / (x + 1) / (y ** (1 / 3))
    jac_x = np.array([dxx, dxy])
    jac_y = np.array([dyx, dyy])
    return [jac_x, jac_y]

def solver(mthd, x0, y0, t0, tMax, beta2):
    nEval = 400
    tEval = np.linspace(t0, tMax, nEval)
    global b2
    b2 = beta2
    if mthd in ['Radau', 'BDF', 'LSODA']:
        nSol = solve_ivp(f, [t0, tMax], [x0, y0], method=mthd, t_eval=tEval, jac=jacf)
    else:
        nSol = solve_ivp(f, [t0, tMax], [x0, y0], method=mthd)
    tSol = nSol.t
    xySol = nSol.y

    with open('ode.txt', 'wt') as w:
        w.write('{0:^7} \t {1:^10} \t {2:^10} \n'.format('t', 'x', 'y'))
        for index in range(len(tSol)):
            w.write('{0:7.3f} \t {1:10.4e} \t {2:10.4e} \n'.format(tSol[index], *xySol[:,index]))
    return tSol, xySol


def solveplotXorY(axis, mthd, x0, y0, xmax, ymax, t0, tMax, beta2):
    tSol, xySol = solver(mthd, x0, y0, t0, tMax, beta2)
    if axis=='x':
        plt.plot(tSol, xySol[0,:], label='x(t)')
        plt.ylabel('x')
        plt.ylim(0, xmax)
    else:
        plt.plot(tSol, xySol[1,:], label='y(t)')
        plt.ylabel('y')
        plt.ylim(0, ymax)
    plt.legend(loc='best')
    plt.xlabel('t')


def solveplotXY(mthd, x0, y0, xmax, ymax, t0, tMax, beta2):
    tSol, xySol = solver(mthd, x0, y0, t0, tMax, beta2)
    plt.plot(xySol[0,:], xySol[1,:], label="({},{})".format(x0, y0))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.legend()


def plotXandYforB(method, x_0, y_0, t_max):
    plt.subplot(3, 2, 1)
    solveplotXorY('x', method, x_0, y_0, 0, t_max, 3)
    plt.title('x(t) for b2=3')
    plt.subplot(3, 2, 2)
    solveplotXorY('y', method, x_0, y_0, 0, t_max, 3)
    plt.title('y(t) for b2=3')

    plt.subplot(3, 2, 3)
    solveplotXorY('x', method, x_0, y_0, 0, 40, 3.48)
    plt.title('x(t) for b2=3.48')
    plt.subplot(3, 2, 4)
    solveplotXorY('y', method, x_0, y_0, 0, 40, 3.48)
    plt.title('y(t) for b2=3.48')

    plt.subplot(3, 2, 5)
    solveplotXorY('x', method, x_0, y_0, 0, 200, 5)
    plt.title('x(t) for b2=5')
    plt.subplot(3, 2, 6)
    solveplotXorY('y', method, x_0, y_0, 0, 200, 5)
    plt.title('y(t) for b2=5')


def vectorfield():
    ymax = 20
    xmax = 3
    nb_points = 20
    X = np.linspace(0, xmax, nb_points)
    Y = np.linspace(0, ymax, nb_points)
    x, y = np.meshgrid(X, Y)
    DX, DY = f(0, [x, y])
    # for j in range(nb_points):
    #     for i in range(nb_points):
    #         DX[j,i], DY[j,i] = f(0, [x[i], y[j]])

    M = np.hypot(DY, DX)
    M[M==0] = 1
    DX /= M
    DY /= M
    Q = plt.quiver(x, y, DX, DY, M, pivot='mid', cmap=plt.cm.plasma)
    plt.grid()
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.title("Векторное поле при b2=" + str(b2))
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.streamplot(x, y, DX, DY)


def stream():
    ymax = 20
    xmax = 3
    nb_points = 20
    X = np.linspace(0, xmax, nb_points)
    Y = np.linspace(0, ymax, nb_points)
    x, y = np.meshgrid(X, Y)
    DX, DY = f(0, [x, y])
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.title("b2=" + str(b2))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.streamplot(x, y, DX, DY)


def showsol(x0, y0, xmax, ymax, tmax, color='blue'):
    plt.subplot(1, 3, 1)
    plt.title("x0="+str(x0)+", y0="+str(y0))
    solveplotXY(method, x0, y0, xmax, ymax, 0, tmax, b2)
    plt.gca().get_lines()[0].set_color(color)
    plt.subplot(1, 3, 2)
    solveplotXorY('x', method, x0, y0, xmax, ymax, 0, tmax, b2)
    plt.gca().get_lines()[0].set_color(color)
    plt.subplot(1, 3, 3)
    solveplotXorY('y', method, x0, y0, xmax, ymax, 0, tmax, b2)
    plt.gca().get_lines()[0].set_color(color)


method="Radau"
# b2 = 3.48
# x0 = 0.5
# y0 = 3
# t0 = 0
# tMax = 20
# nEval = 400
# tEval = np.linspace(t0, tMax, nEval)
# tic1 = time()
# if mthd in ['Radau', 'BDF', 'LSODA']:
#     nSol = solve_ivp(f, [t0, tMax], [x0, y0], method=mthd, t_eval=tEval, jac=jacf)
# else:
#     nSol = solve_ivp(f, [t0, tMax], [x0, y0], method=mthd)
# tic2 = time()
# print('Метод - {0}\nВремя расчетов (сек.):{1}\nКоличество точек:{2}'.format(mthd, tic2 - tic1, len(nSol.t)))

# b2 = 3
# fig = plt.figure(1)
# # vectorfield()
# stream()
# solveplotXY(method, 0.5, 5, 3, 20, 0, 20, b2)
# solveplotXY(method, 2.9, 5, 3, 20, 0, 20, b2)
# # solveplotXY(method, 2, 1, 3, 20, 0, 20, b2)
# #
# fig = plt.figure(2, constrained_layout=True)
# showsol(2.9, 5, 4, 20, 20)
# fig = plt.figure(3, constrained_layout=True)
# showsol(0.5, 5, 5, 20, 20)

# b2 = 3.48
# ymax = 20
# fig = plt.figure(1)
# # vectorfield()
# stream()
# solveplotXY(method, 0.5, 4., 3, ymax, 0, 600, b2)
# solveplotXY(method, 2.9, 2, 3, ymax, 0, 20, b2)
# # solveplotXY(method, 1.2, 5.0, 3, ymax, 0, 100, b2)
# # solveplotXY(method, 1.2, 3.3, 3, ymax, 0, 20, b2)
# # solveplotXY(method, 2.4, 1.5, 3, ymax, 0, 20, b2)
# solveplotXY(method, 1.5, 7, 3, ymax, 0, 60, b2) # уходит вверх
# solveplotXY(method, 1.5, 6.5, 3, ymax, 0, 600, b2) # уходит в фокус
# fig = plt.figure(2, constrained_layout=True)
# showsol(1.5, 6.5, 3, 20, 100)
# fig = plt.figure(3, constrained_layout=True)
# showsol(1.5, 7., 3, 20, 60)

b2 = 5
fig = plt.figure(1)
# vectorfield()
stream()
solveplotXY(method, 0.5, 4., 3, 20, 0, 60, b2)
solveplotXY(method, 1.2, 7.3, 3, 20, 0, 20, b2)
solveplotXY(method, 1.2, 5.0, 3, 20, 0, 20, b2)
solveplotXY(method, 1.2, 3.3, 3, 20, 0, 20, b2)
solveplotXY(method, 2, 7.3, 3, 20, 0, 20, b2)
solveplotXY(method, 2.6, 14.0, 3, 20, 0, 20, b2)
fig = plt.figure(2, constrained_layout=True)
showsol(1, 2, 3, 20, 200)
fig = plt.figure(6, constrained_layout=True)
showsol(1, 9, 3, 20, 200)


plt.show()
