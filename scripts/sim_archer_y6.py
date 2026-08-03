#!/usr/bin/env python3
"""sim_archer_y6.py — Archer Y6 joint cycling smoke test.

Usage::

    source shells/isaaclab2.2.1_env.sh
    python pkg_hex/hex_sim_isaaclab_wrapper/scripts/sim_archer_y6.py --steps 500

Simulation parameters (device, headless, num_envs, grip_type) are configured
directly in HexRobotSimArcherY6Params below.
"""

import argparse
import time

import numpy as np

from hex_sim_isaaclab_wrapper import HexRobotSimArcherY6, HexRobotSimArcherY6Params


def main():
    parser = argparse.ArgumentParser(description="Archer Y6 sim smoke test")
    parser.add_argument("--steps", type=int, default=0, help="Steps (0=infinite)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    args = parser.parse_args()

    # Configure simulation parameters here
    params = HexRobotSimArcherY6Params(
        headless=args.headless,
        num_envs=1,
        grip_type="gr100",
        ctrl_rate = 100.0
    )

    robot = HexRobotSimArcherY6(params)
    robot.start()
    print("[INFO]: Setup complete, starting joint cycling.", flush=True)
    
    arm_state = None
    grip_state = None

    count = 0
    total = 0
    while robot.is_working():
        if args.steps > 0 and total >= args.steps:
            break
        total += 1

        if count < 200:
            target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            tar_pos = [0.3,0.0,0.6]
            tau = [0.5,0.5]
        elif count < 400:
            target = [0.0, -1.5, 3.0, 0.0, 0.0, 0.0]
            tar_pos = [0.3,0.0,0.2]
            
        else:
            count = 0

        robot.set_arm_mit_cmd({
            "mit_kp": np.full(6, 0.0),
            "mit_kd": np.full(6, 0.0),
            "target": target,
        })

        # robot.set_arm_pos_cmd({
        #     "jnt_pos": target,
        #     "lim_vel": np.full(6, 100.0),   # smooth interpolated trajectory
        # })
        # robot.set_arm_pose_cmd({
        #     "pose_pos": tar_pos,
        #     "pose_quat": np.asarray([1.0,0.0,0.0,0.0]),   # smooth interpolated trajectory
        #     "lim_vel": np.full(6, 100.0)
        # })
        
        robot.set_grip_pos_cmd({"jnt_pos": tau})

        robot.step()

        arm_state = robot.get_arm_state()
        grip_state = robot.get_grip_state()
        
        if arm_state is not None and total % 50 == 0:
            print(f"[arm] {total}: pos={np.round(arm_state.arm_state.jnt.position, 3)}",
                  flush=True)

        count += 1
        time.sleep(1.0 / params.ctrl_rate)

    robot.stop()
    print(f"[INFO]: Done ({total} steps).")


if __name__ == "__main__":
    main()
