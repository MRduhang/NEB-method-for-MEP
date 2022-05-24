import numpy as np
import matplotlib.pyplot as plt


def get_gaussian(x, y, z, x0, y0, z0):
    return np.exp(-(x - x0) ** 2 - (y - y0) ** 2 - (z - z0) ** 2)


def get_data(x, y, z):
    data = get_gaussian(x, y, z, 0.1, -0.1, -0.1) - get_gaussian(x, y, z, -0.5, -0.5, -0.5) - get_gaussian(x, y, z, 0.5,
                                                                                                           0.5, 0.5)
    return data


"""
step1.找极小值点
"""


# 计算某一点的梯度
def compute_gradient(point, epsilon=1e-2):
    x, y, z = point[0], point[1], point[2]
    # dx = (f(x + h) - f(x - h)) / 2*h
    grad_x = (get_data(x + epsilon, y, z) - get_data(x - epsilon, y, z)) / (2 * epsilon)
    grad_y = (get_data(x, y + epsilon, z) - get_data(x, y - epsilon, z)) / (2 * epsilon)
    grad_z = (get_data(x, y, z + epsilon) - get_data(x, y, z - epsilon)) / (2 * epsilon)
    grad = np.array([grad_x, grad_y, grad_z])

    return grad


# 给定初始化点，使用割线法分别分析x,y,z的梯度，并对其进行梯度下降
def gradient_descent(point, descent_rate=5e-1, iters=100):
    # 梯度下降
    for iter in range(iters):
        # 求给定点的梯度
        grad = compute_gradient(point)
        # 更新参数
        for i in range(len(point)):
            point[i] = point[i] - grad[i] * descent_rate

    return point


# 寻找[0.5, 0.5, 0.5]附近的局部极小值点
min_a = np.array([0.5, 0.5, 0.5])
min_a = gradient_descent(min_a, iters=100)
print(f'在[0.5, 0.5, 0.5]附近：\n\t局部极小点为: {min_a}')
print(f'\t极值 = {get_data(min_a[0], min_a[1], min_a[2])}\n')

# 寻找[-0.5, -0.5, -0.5]附近的局部极小值点
min_b = np.array([-0.5, -0.5, -0.5])
min_b = gradient_descent(min_b, iters=100)
print(f'在[-0.5, -0.5, -0.5]附近：\n\t局部极小点为: {min_b}')
print(f'\t极值 = {get_data(min_b[0], min_b[1], min_b[2])}\n')

"""
step2.通过NEB方法计算鞍点
"""


# 点集2d可视化
def plot_2Dpoints(points, title='Points Path'):
    ax = plt.subplot()
    ax.scatter(points[:, 0], points[:, 1], c='r')
    ax.plot(points[:, 0], points[:, 1], c='r')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


# 点集3d可视化
def plot_3Dpoints(points, title='Points Path', saddle_point=np.zeros([1, 3])):
    ax = plt.subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], c='r')
    if saddle_point.any() != 0:
        ax.scatter(*saddle_point, c='k', s=80, marker='*')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# 定义创建弹性带函数
def creat_elastic_bond(start_point, end_point, n):
    '''
    从起点到终点创建一条由n个点构成的弹性带
    params:
        start_point ———— 起点
        end_point ———— 终点
        n ———— 由多少点构成
    return:
        points ———— 构成弹性带的点集
    '''
    points = np.zeros([n, 3])
    for i in range(0, n):
        points[i] = (end_point - start_point) / (n - 1) * i + start_point

    return points


# 定义计算两点之间距离的函数
def compute_dist(p1, p2):
    return np.sqrt(np.sum(np.power((p2 - p1), 2)))


# 定义计算两极小值点间的平均距离的函数
def compute_dist_avg(start_point, end_point, n):
    '''
    params:
        start_point ———— 起点
        end_point ———— 终点
        n ———— 弹性带由多少点构成
    '''
    return compute_dist(start_point, end_point) / (n - 1)


# 定义计算两点之间弹力大小的函数
def compute_force(dist, dist_avg, elastic_rate):
    return (dist - dist_avg) * elastic_rate


# 定义计算两点之间弹力方向的函数
def compute_direction(p1, p2, dist):
    return (p2 - p1) / dist


# 定义计算某一点point的所受弹力大小和方向的函数
def elastic_force(point_last, point, point_next, dist_avg, elastic_rate=5e-1):
    '''
        params:
            point_last ———— 给定点
            point ———— 给定点的上一个点
            point_next ———— 给定点的下一个点
            dist_avg ———— 两个极小值点间的平均距离
            elastic_rate ———— 弹性系数
        return:
            points ———— 由点组成的MEP弹性带
        '''
    # 计算给定点与它上一个点的弹力及其方向
    dist_last = compute_dist(point_last, point)
    force_last = compute_force(dist_last, dist_avg, elastic_rate)
    direction_last = compute_direction(point_last, point, dist_last)

    # 计算给定点与它下一个点的弹力及其方向
    dist_next = compute_dist(point, point_next)
    force_next = compute_force(dist_next, dist_avg, elastic_rate)
    direction_next = compute_direction(point_next, point, dist_next)

    # 计算所求点的所受合力及其方向
    force = force_last * direction_last + force_next * direction_next
    if np.sum(force ** 2) < 1e-10:
        direction = np.zeros_like(force)
    else:
        direction = force / np.sqrt(np.sum(force ** 2))

    return force, direction


# 定义NEB方法函数(求最小能量路径)
def neb(start_point, end_point, n, elastic_rate, descent_rate=5e-3, iters=1000, plot_origin=False, plot_mep=False):
    '''
    params:
        start_point ———— 起点
        end_point ———— 终点
        n ———— 弹性带由多少点构成
        elastic_rate ———— 弹性带的弹性系数
        descent_rate ————弹性带的滑动速率
        iters ———— 迭代次数
        plot ———— 是否可视化由点组成的弹性带
    return:
        points ———— 由点组成的MEP弹性带
    '''
    # 创建初始弹性带
    points = creat_elastic_bond(start_point, end_point, n)

    # 是否可视化初始弹性带
    print(f'初始弹性带为：\n{points}\n')
    if plot_origin == True:
        plot_2Dpoints(points, 'Origin Elastic Bond')
        plot_3Dpoints(points, 'Origin Elastic Bond')

    # 计算两极小值点间的平均距离
    dist_avg = compute_dist_avg(start_point, end_point, n)

    # 迭代计算最小能量路径MEP
    for iter in range(iters):
        for i in range(1, n - 1):  # 起点和终点固定不变
            e_force, direction = elastic_force(points[i - 1], points[i], points[i + 1], dist_avg, elastic_rate)
            e_force = np.dot(e_force, direction) * direction
            grad_force = compute_gradient(points[i])
            grad_force = grad_force - np.dot(grad_force, direction) * direction
            points[i] -= (e_force + grad_force) * descent_rate

    # 是否可视化MEP弹性带
    print(f'最小能量路径弹性带为：\n{points}\n')
    if plot_mep == True:
        plot_2Dpoints(points, 'MEP Elastic Bond')
        plot_3Dpoints(points, 'MEP Elastic Bond')

    return points


# 定义找出最小能量路径中的鞍点的函数
def find_saddle_point(points, plot_saddle_point=False):
    # 函数值最大的点即为鞍点
    points_data = get_data(points[:, 0], points[:, 1], points[:, 2])
    saddle_point_index = np.squeeze(np.where(points_data == max(points_data)))
    saddle_point = points[saddle_point_index]
    saddle_point_values = get_data(*saddle_point)

    # 打印鞍点位置及其函数值
    print(f'鞍点位置为：{saddle_point}, 其函数值 = {saddle_point_values}\n')
    if plot_saddle_point == True:
        plot_3Dpoints(points, 'MEP Elastic Bond', saddle_point=saddle_point)

    return saddle_point, saddle_point_values


new_points = neb(min_a, min_b, n=11,
                 elastic_rate=1.5, descent_rate=1e-1,
                 iters=2000, plot_origin=False, plot_mep=False)

saddle_point, saddle_point_values = find_saddle_point(new_points, plot_saddle_point=True)
