import numpy as np
from plyImporter import PlyImporter
import taichi as ti


ply3 = PlyImporter("long.ply")  # import point cloud file
dim, n_grid, steps, dt, res = 3, 64, 25, 2e-4, 500  # 更友好的配置
#dim, n_grid, steps, dt, res = 3, 128, 8, 1e-4, 720  # 维度, 网格数, 模拟帧率, 时间步长, 分辨率

n_particles = ply3.get_count()
print(n_particles)
dx, inv_dx = 1 / n_grid, float(n_grid)  # 格点（单个网格的中心）
p_vol, p_rho = (dx * 0.5) ** 3, 1
p_mass = p_vol * p_rho  # 粒子质量
gravity = 9.8  # 重力
bound = 3  # 边界值
E, nu = 400, 0.2  # 杨氏模量和泊松比
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # 拉梅参数

x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # 位置向量数组
v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # 速度向量数组
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # 仿射速度矩阵数组
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # 变形梯度矩阵数组
Jp = ti.field(dtype=ti.f32, shape=n_particles)  # 塑性变形数组
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid,)*dim)  # 网格节点动量数组
grid_m = ti.field(dtype=ti.f32, shape=(n_grid,)*dim)  # 网格节点质量数组
material = ti.field(dtype=ti.int32, shape=n_particles)  # 粒子材质数组
neighbour = (3, ) * dim

@ti.kernel
def substep():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])  # 重置每个网格节点的速度
        grid_m[I] = 0  # 重置每个网格节点的质量
    ti.loop_config(block_dim=n_grid)
    # ti.block_dim(n_grid)

    for p in x:  # 把粒子状态更新到网格节点 (P2G)
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]  # 变形梯度更新
        h = ti.exp(10 * (1 - Jp[p]))  # 硬化系数
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # 液体
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0  # J是粒子的体积变化

        for d in ti.static(range(3)):  # 处理材质
            new_sig = sig[d, d]
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = J
            F[p] = new_F  # 重置变形梯度以避免数值不稳定
        elif material[p] == 2:
            F[p] = U @ sig @ V.transpose()  # 塑性后重建弹性变形梯度

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + p_mass * C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for I in ti.grouped(grid_m):  # (P2G) 这一步是为了把网格内粒子的质量加权平均赋加到网格上
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I]
        grid_v[I][1] -= dt * gravity
        cond = (I < bound) & (grid_v[I] < 0) | (I > n_grid - bound) & (grid_v[I] > 0)  # 边界条件
        grid_v[I] = ti.select(cond, 0, grid_v[I])
    ti.loop_config(block_dim=n_grid)

    for p in x:  # 从网格节点获取状态到粒子 (G2P)
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(v[p])
        new_C = ti.zero(C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]



@ti.kernel
def copy_material(np_x: ti.types.ndarray(), input_x: ti.template()):
    for i in x:
        np_x[i] = input_x[i]

@ti.kernel
def copy_color(np_c: ti.types.ndarray(), input_c: ti.types.ndarray()):
    for i in x:
        np_c[i] = input_c[i]

@ti.kernel
def initialize():
    for i in range(n_particles):  # 初始化粒子的位置
        material[i] = 0
        v[i] = ti.Matrix([0, 0, 0])  # 初始化速度
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 初始化形变梯度
        Jp[i] = 1  #初始化形变


def T(a):  # 视角投影变换
    phi, theta = np.radians(28), np.radians(30)  # 左右角度， 上下角度

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.5


def save_ply(frame1):
    # series_prefix = "dragon_solved.ply"
    series_prefix = "long_solved.ply"
    num_vertices = n_particles
    np_pos = np.reshape(x.to_numpy(), (num_vertices, 3))
    writer = ti.tools.np2ply.PLYWriter(num_vertices=num_vertices)
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    writer.export_frame_ascii(frame1+1, series_prefix)