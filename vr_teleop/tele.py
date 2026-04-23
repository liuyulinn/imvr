"""
scripts for teleop in simulation
"""

import faulthandler

faulthandler.enable()

import argparse
import time

import sapien
import tqdm

sapien.render.enable_vr()

from vr_teleop.envs.base_env import EnvBuilderConfig, env_builder

from vr_teleop.modules.tele import TeleFlags
from vr_teleop.modules.teleconfig import TeleopConfig

from vr_teleop.modules.vr.initialize import create_viewer


parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="configs/xarm_gripper/env.yaml", help="env config file")
parser.add_argument("--tele", type=str, default="configs/xarm_gripper/tele.yaml", help="teleop config file")
parser.add_argument(
    "--log_time", type=float, default=10, help="time interval to print render frequency and control frequency"
)
parser.add_argument("--render_freq", type=int, default=60, help="render frequency, 60Hz for smooth user experience")
args = parser.parse_args()


def main():
    cfg = EnvBuilderConfig.from_yaml(args.env)
    env_cfg = cfg.get_cfg()  # type: ignore
    env = env_builder.create(cfg._type_)(1, env_cfg)  # type: ignore

    config = env.config

    env.reset()

    scene = env.get_cpu_scene()

    teleopconfig = TeleopConfig.from_yaml(args.tele)

    viewer = create_viewer(env, teleopconfig.viewer, env_cfg)

    teleop = teleopconfig.create_teleop_modules(env, scene, viewer)  # type: ignore

    env.set_viewer(viewer)  # type: ignore

    num_trajs = 0
    num_fail_trajs = 0
    start_time = None
    epside_time = 0

    next_control_time = time.time()
    next_render_time = time.time()

    render_count = 0
    control_count = 0
    last_log_time = time.time()

    try:
        for iters in tqdm.trange(1000000000):
            cur_time = time.time()

            cur_qposs = env.agent_qpos

            root_poses = env.agent_root_pose

            if cur_time - next_render_time > 0:
                viewer.render()
                next_render_time += 1 / args.render_freq
                render_count += 1

            if cur_time > next_control_time:
                _flags, actions = teleop.step(cur_qposs, root_poses)

                if _flags == TeleFlags.IK_SUCCESS:
                    start_time = time.time() if start_time is None else start_time
                    _state, _reward, _done, _truncated, info = env.step(actions)  # type: ignore()

                    if _done:  # or _truncated:
                        env.reset(save=True)
                        teleop.reset()
                        num_trajs += 1
                        print(f"Collected {num_trajs} trajectories")

                elif _flags == TeleFlags.DONE:
                    env.reset(save=True)
                    teleop.reset()  # type: ignore

                    num_trajs += 1
                    print(f"Collected {num_trajs} trajectories")

                elif _flags == TeleFlags.CLEAR_ERROR:
                    for i in range(len(actions)):
                        env.clean_warning_error(i)

                elif _flags == TeleFlags.RESET:
                    env.reset(save=False)
                    teleop.reset()  # type: ignore

                    num_fail_trajs += 1

                next_control_time += 1 / env.control_freq
                control_count += 1

            if cur_time - last_log_time >= args.log_time:
                print(
                    f"[Loop Rates] Render FPS: {render_count / args.log_time}, Control Hz: {control_count / args.log_time}"
                )
                render_count = 0
                control_count = 0
                last_log_time = cur_time

    except KeyboardInterrupt:
        del teleop
        del env


if __name__ == "__main__":
    main()
