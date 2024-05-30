import taichi as ti
import numpy as np


dim = 3
n_grid = 64
grid_size = (n_grid,) * dim
# neighbourRadius = 1.5
neighbours = (3,) * dim
frameSteps = 25
dt = 1e-4
space_side_length = 1.0
# simulate in 1 x 1 x 1 cube
dx = space_side_length / n_grid  # grid spacing
inv_dx = 1.0 / dx
# region material point properties
p_density = 1.0
# make all points have the same assumed volume
p_vol = (dx * 0.1)**dim
p_mass = p_density * p_vol
# endregion
gravity = [0.0, -9.8, 0.0]
# bound constraint
lowerBound = [3, 3, 10]
upperBound = [n_grid - 3, n_grid - 40, n_grid - 20]
num_blocks = 6
block_particles = 1000
fluid_particles = 100000
num_particles = num_blocks * block_particles + fluid_particles

E = 1000  # Young's modulus, ratio of linear stress to linear strain
nu = 0.3  # Poisson's ratio, negative ratio of transverse strain to axial strain
# For isotropic materials (those having identical properties in all directions)
# shear modulus, derived from the general three-dimensional stress-strain relationships
G = E / (2 * (1 + nu))  # E = 2G(1 + nu)
# It provides a way to characterize the material's volumetric elasticity, i.e., how it responds to pressure or changes in volume.
Lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu))  # Lame's first parameter
# region particles' information
p_pos = ti.Vector.field(dim, dtype=float, shape=num_particles)
p_vel = ti.Vector.field(dim, dtype=float, shape=num_particles)
# affine velocity field of each particle
# an affine velocity field is a way of describing how the velocity of each particle changes depending on its position in a linear manner.
# the assumption of a linear or affine relationship between position and velocity is a reasonable approximation.
p_C = ti.Matrix.field(dim, dim, dtype=float, shape=num_particles)
# deformation gradient
# for calculating local strains and stresses at each material point in the body
p_de_grad = ti.Matrix.field(dim, dim, dtype=float, shape=num_particles)
# Determinant of the deformation gradient
# p_J = ti.field(dtype=float, shape=num_particles)
# render colors of each particle
p_colors = ti.Vector.field(4, dtype=float, shape=num_particles)
p_material = ti.field(dtype=int, shape=num_particles)
# particles may not be all simulated sometimes
p_active = ti.field(dtype=bool, shape=num_particles)
# endregion
# auxiliary grid in material point method (MPM)
# grid momentum/velocity
grid_vel = ti.Vector.field(dim, dtype=float, shape=grid_size)
grid_mass = ti.field(dtype=float, shape=grid_size)
# material type enumeration
FLUID = 0
BLOCK = 1


@ti.kernel
def substep():
    # initialize / reset grid
    for grid_index in ti.grouped(grid_mass):
        # grid_index is a 3D vector
        grid_mass[grid_index] = 0
        grid_vel[grid_index] = ti.zero(grid_vel[grid_index])
    ti.loop_config(block_dim=n_grid)
    # Particle state update and scatter to grid (P2G)
    for i in p_pos:
        if not p_active[i]:
            continue
        # deformation gradient update
        p_de_grad[i] = (ti.Matrix.identity(float, dim) +
                        dt * p_C[i]) @ p_de_grad[i]
        h = 1.0  # no volume change
        if p_material[i] == BLOCK:
            h = hardeness  # compressible, soft
        mu = G * h
        la = Lambda * h
        if p_material[i] == FLUID:
            mu = viscosity
            la = la * (1 - viscosity)
        # singular value decomposition (A=USV^T)
        U, sigs, V = ti.svd(p_de_grad[i])
        J = 1.0
        for d in ti.static(range(dim)):
            sig = sigs[d, d]
            J *= sig  # determinant of the diagonal matrix sigs
        if p_material[i] == FLUID:
            # reset deformation gradient to avoid numerical instability
            temp = ti.Matrix.identity(float, dim)
            temp[0, 0] = J
            p_de_grad[i] = temp
        stress = 2 * mu * (p_de_grad[i] - U @ V.transpose()) @ p_de_grad[i].transpose(
        ) + ti.Matrix.identity(float, dim) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx**2) * stress
        affine = stress + p_mass * p_C[i]
        # grid coordinate
        grid_coord = p_pos[i] * inv_dx
        # base is the left-bottom-back corner of the grid cell 3 x 3 x 3
        base = int(grid_coord - 0.5)
        p_offset = grid_coord - base
        # Quadratic kernels
        w = [0.5 * (1.5 - p_offset) ** 2, 0.75 - (p_offset - 1)
             ** 2, 0.5 * (p_offset - 0.5) ** 2]
        # Loop over 3x3 grid node neighborhood
        for g_offset in ti.static(ti.grouped(ti.ndrange(*neighbours))):
            # grid node coordinate
            g_coord = base + g_offset
            weight = 1.0
            displacement = (g_offset - p_offset) * dx
            for d in ti.static(range(dim)):
                weight *= w[g_offset[d]][d]
            # grid infomation at each node is accumulated by iterating over all particles
            # grid momentum update
            grid_vel[g_coord] += weight * \
                (p_mass * p_vel[i] + affine @ displacement)
            # grid mass update
            grid_mass[g_coord] += weight * p_mass
    # ti.loop_config(block_dim=n_grid)
    for i in ti.grouped(grid_mass):
        if grid_mass[i] > 0:
            # velocity = momentum / mass
            grid_vel[i] = grid_vel[i] / grid_mass[i]
        # gravity
        grid_vel[i] += dt * ti.Vector(gravity)
        # a cubiod boundary
        # set x,y,z velocity to zero if the grid node is on the boundary while the velocity is pointing outside of the boundary along the axis
        IsEscapingBoundary = (i < lowerBound) & (grid_vel[i] < 0) | (
            i > upperBound) & (grid_vel[i] > 0)
        grid_vel[i] = ti.select(IsEscapingBoundary, 0, grid_vel[i])
    ti.loop_config(block_dim=n_grid)
    # grid to particle (G2P)
    for i in p_pos:
        if not p_active[i]:
            continue
        # grid coordinate
        grid_coord = p_pos[i] * inv_dx
        # base is the left-bottom-back corner of the grid cell 3 x 3 x 3
        base = int(grid_coord - 0.5)
        p_offset = grid_coord - base
        # Quadratic kernels
        w = [0.5 * (1.5 - p_offset) ** 2, 0.75 - (p_offset - 1)
             ** 2, 0.5 * (p_offset - 0.5) ** 2]
        updated_vel = ti.zero(p_vel[i])
        updated_C = ti.zero(p_C[i])
        # Loop over 3x3 grid node neighborhood
        for g_offset in ti.static(ti.grouped(ti.ndrange(*neighbours))):
            # grid node coordinate
            g_coord = base + g_offset
            weight = 1.0
            dpos = (g_offset - p_offset) * dx
            for d in ti.static(range(dim)):
                weight *= w[g_offset[d]][d]
            updated_vel += weight * grid_vel[g_coord]
            updated_C += 4 * weight * \
                grid_vel[g_coord].outer_product(dpos) / dx**2
        p_vel[i] = updated_vel
        p_pos[i] += dt * p_vel[i]
        p_C[i] = updated_C


@ti.kernel
def prepare_particles_out_of_window():
    for i in p_active:
        p_active[i] = False
        # don't show in the window initially
        p_pos[i] = ti.Vector([900000.0, 999999.0, 999999.0])
        p_de_grad[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        p_C[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        p_vel[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def spawnBlock(start: int, end: int, x_size: float, y_size: float, z_size: float, x_begin: float, y_begin: float, z_begin: float):
    color = ti.Vector([ti.random(), ti.random(), ti.random(), 1.0])
    for i in range(start, end):
        p_pos[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector([x_size, y_size, z_size]) + ti.Vector(
            [x_begin, y_begin, z_begin]
        )
        p_de_grad[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        p_vel[i] = ti.Vector([0.0, 0.0, 0.0])
        p_material[i] = BLOCK
        p_active[i] = True
        p_colors[i] = color


@ti.kernel
def spawnFluid(start: int, end: int, x_size: float, y_size: float, z_size: float, x_begin: float, y_begin: float, z_begin: float):
    color = ti.Vector([0.0, 0.64, 0.89, 0.2])
    for i in range(start, end):
        p_pos[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector([x_size, y_size, z_size]) + ti.Vector(
            [x_begin, y_begin, z_begin]
        )
        p_de_grad[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        p_vel[i] = ti.Vector([0.0, -3.0, 0.0])
        p_material[i] = FLUID
        p_active[i] = True
        p_colors[i] = color


def set_scene(vols):
    prepare_particles_out_of_window()
    particle_start = fluid_particles
    for i, v in enumerate(vols):
        if isinstance(v, Block):
            if v.material == BLOCK:
                particle_end = particle_start + block_particles
                spawnBlock(particle_start, particle_end, *v.size, *v.minimum)
                particle_start = particle_end
        else:
            raise Exception("Unknown type")
    print("Spawned", particle_start, "particles")


class Block:
    def __init__(self, minimum, size, material):
        self.minimum = minimum
        # size is a ti.Vector has 3 properties: x, y, z
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material


scenes = [
    [
        Block(ti.Vector([0.2, 0.0, 0.25]),  # 1
              ti.Vector([0.05, 0.05, 0.05]), BLOCK),
        Block(ti.Vector([0.5, 0.0, 0.4]),  # 2
              ti.Vector([0.05, 0.05, 0.05]), BLOCK),
        Block(ti.Vector([0.5, 0.08, 0.4]),  # 3
              ti.Vector([0.05, 0.05, 0.05]), BLOCK),
        Block(ti.Vector([0.4, 0.25, 0.4]),  # 4
              ti.Vector([0.05, 0.05, 0.05]), BLOCK),
        Block(ti.Vector([0.4, 0.15, 0.6]),  # 5
              ti.Vector([0.05, 0.05, 0.05]), BLOCK),
        Block(ti.Vector([0.6, 0.05, 0.6]),  # 6
              ti.Vector([0.05, 0.05, 0.05]), BLOCK),
    ],
]

# settings
current_scene = 0
particles_radius = 0.003
viscosity = 0.0
hardeness = 0.6
frames = 300

res = (1080, 720)
window = ti.ui.Window("Fluidflow", res, vsync=True)

canvas = window.get_canvas()
canvas.set_background_color((0.4, 0.4, 0.4))
gui = window.get_gui()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
# camera.position(1, 1.5, 1.95)
camera.position(1.2, 1.98, 2.53)
# camera.position(0.8, 1.02, 1.37)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(20)


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.particles(p_pos, per_vertex_color=p_colors, radius=particles_radius)
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    canvas.scene(scene)