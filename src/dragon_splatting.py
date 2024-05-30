import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

from DreaMPM import simplified
from tqdm import tqdm


simplified.x.from_numpy(simplified.ply3.get_array())
simplified.initialize()
for frame in tqdm(range(120)):
    for s in range(simplified.steps):
        simplified.substep()

    simplified.save_ply(frame)

# gui = ti.GUI("Dragon Splatting", simplified.res, background_color=0x112F41)
# while gui.running and not gui.get_event(gui.ESCAPE):
#     for s in range(simplified.steps):
#         simplified.substep()  # 更新模拟帧
#
#     pos = simplified.x.to_numpy()  # 转换位置为numpy数组，以便GUI调用
#
#     colors = np.array([0x068599, 0xFF8888, 0xEEEEF0], dtype=np.uint32)  # 材质颜色
#     np_color = np.ndarray((simplified.n_particles,), dtype=np.uint32)  # 粒子颜色数组
#     simplified.copy_color(np_color, colors)
#
#     np_material = np.ndarray((simplified.n_particles,), dtype=np.uint32)  # 粒子材质索引数组
#     simplified.copy_material(np_material, simplified.material)
#
#     gui.circles(simplified.T(pos), radius=1.6, color=np_color[np_material])
#     gui.show()
