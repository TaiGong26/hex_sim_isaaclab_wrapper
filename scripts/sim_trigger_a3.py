#!/usr/bin/env python3
"""sim_trigger_a3.py — Trigger A3 chassis MIT smoke test.

Usage::

    source shells/isaaclab2.1.1_env.sh
    python pkg_hex/hex_sim_isaaclab_wrapper/scripts/sim_trigger_a3.py --headless --steps 300

Sends a constant MIT command (jnt_pos=0, jnt_vel=full(1.0), mit_kd=full(3.0))
to all 3 omni-wheels and prints chassis state every 50 steps.
"""

import argparse
import time

import numpy as np

from hex_sim_isaaclab_wrapper import HexRobotSimTriggerA3, HexRobotSimTriggerA3Params


def main():
    parser = argparse.ArgumentParser(description="Trigger A3 chassis sim smoke test")
    parser.add_argument("--steps", type=int, default=0, help="Steps (0=infinite)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    args = parser.parse_args()

    params = HexRobotSimTriggerA3Params(
        isaac_headless=args.headless,
        ctrl_rate=500.0,
        torch_device="cuda:0",
    )
    robot = HexRobotSimTriggerA3(params)
    robot.start()
    print("[INFO]: Setup complete, starting A3 MIT test.", flush=True)

    dof = 3
    jnt_vel = np.full(dof, 1.0)   # all 3 omni-wheels forward

    count = 0
    while robot.is_working():
        if args.steps > 0 and count >= args.steps:
            break
        robot.set_chs_mit_cmd({
            "jnt_pos": np.zeros(dof),
            "jnt_vel": jnt_vel,
            "mit_tau": np.zeros(dof),
            "mit_kp": np.zeros(dof),   # A3 config kp=0; explicit zero → no pos restoring force
            "mit_kd": np.full(dof, 3.0),
        })
        robot.step()
        if count % 50 == 0:
            st = robot.get_chassis_state()
            if st is not None:
                cs = st.chs_state
                print(f"[a3] {count}: jnt_len={len(cs.jnt.position)} "
                      f"odom.pose.position={cs.odom.pose.position} "
                      f"odom.twist.linear={cs.odom.twist.linear}", flush=True)
        count += 1
        time.sleep(1.0 / params.ctrl_rate)

    robot.stop()
    print(f"[INFO]: Done ({count} steps).")


if __name__ == "__main__":
    main()
