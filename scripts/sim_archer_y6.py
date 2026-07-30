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

from hex_sim_isaaclab_wrapper import HexRobotSimArcherY6, HexRobotSimArcherY6Params


def main():
    parser = argparse.ArgumentParser(description="Archer Y6 sim smoke test")
    parser.add_argument("--steps", type=int, default=0, help="Steps (0=infinite)")
    args = parser.parse_args()

    # Configure simulation parameters here
    params = HexRobotSimArcherY6Params(
        device="cuda:0",
        headless=False,
        num_envs=1,
        grip_type="gr100",
    )

    robot = HexRobotSimArcherY6(params)
    robot.start()
    print("[INFO]: Setup complete, starting joint cycling.", flush=True)
    
    arm_state = None
    grip_state = None

    count = 0
    while robot.is_working():
        if args.steps > 0 and count >= args.steps:
            break

        if count < 200:
            target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            tau = [0.5,0.5]
        elif count < 400:
            target = [0.0, -1.5, 3.0, 0.0, 0.0, 0.0]
            tau = [0.0,0.0]
            
        else:
            count = 0

        robot.set_arm_pos_cmd({"jnt_pos": target})
        robot.set_grip_pos_cmd({"jnt_pos": tau})
        
        arm_state = robot.get_arm_state()
        grip_state = robot.get_grip_state()
        
        if arm_state is not None:
            print(f"[arm]: pos={arm_state.arm_state.jnt.position}  "
                  f"vel={arm_state.arm_state.jnt.velocity}")

        if grip_state is not None:
            print(f"[grip]: pos={grip_state.grip_state.jnt.position}  "
                  f"vel={grip_state.grip_state.jnt.velocity}")


    robot.stop()
    print(f"[INFO]: Done ({count} steps).")


if __name__ == "__main__":
    main()
