import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

from DreaMPM import water_flow

def main():
    water_flow.set_scene(water_flow.scenes[water_flow.current_scene])
    start = 0
    end = 500
    outputVideo = False
    result_dir = "./waterflow"
    video_manager = ti.tools.VideoManager(
        output_dir=result_dir, framerate=24, automatic_build=False)
    for i in range(water_flow.frames):
        if end < water_flow.fluid_particles:
            water_flow.spawnFluid(start, end, 0.1, 0.1, 0.25, 0.1, 0.5, 0.15)
            start = end
            end += 500
            water_flow.spawnFluid(start, end, 0.1, 0.1, 0.25, 0.1, 0.5, 0.45)
            start = end
            end += 500
        for _ in range(water_flow.frameSteps):
            water_flow.substep()

        water_flow.render()
        water_flow.window.show()
        if outputVideo:
            pixels = water_flow.window.get_image_buffer_as_numpy()
            video_manager.write_frame(pixels)
    if outputVideo:
        print('Exporting .mp4 and .gif videos...')
        video_manager.make_video(gif=True, mp4=True)


if __name__ == "__main__":
    main()
