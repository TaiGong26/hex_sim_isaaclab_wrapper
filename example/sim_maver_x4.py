#!/usr/bin/env python3
"""sim_maver_x4.py — Maver X4 chassis MIT smoke test.

Example:

```bash
source shells/isaaclab2.1.1_env.sh
python pkg_hex/hex_sim_isaaclab_wrapper/scripts/sim_maver_x4.py --headless --steps 300
```

Sends a constant MIT command (`jnt_pos=0`, `jnt_vel`: drive wheels → 3.0 rad/s,
steering → 0, `mit_kd=full(3.0)`) and prints chassis state every 50 steps.
Joint order is the canonical (ROS2) order
`[joint_wheel1, joint_yaw1, joint_wheel2, joint_yaw2, joint_wheel3, joint_yaw3,
joint_wheel4, joint_yaw4]`: drive = indices [0,2,4,6], steering = [1,3,5,7].
"""

import argparse
import time

import numpy as np

from hex_sim_isaaclab_wrapper import HexRobotSimMaverX4, HexRobotSimMaverX4Params


def main() -> None:
    """Run the Maver X4 MIT smoke test and print chassis state periodically.

    Args:
        --steps: Number of steps to run (0 = infinite).
        --headless: Run without a GUI window.
    """
    parser = argparse.ArgumentParser(description="Maver X4 chassis sim smoke test")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    args = parser.parse_args()

    params = HexRobotSimMaverX4Params(
        isaac_headless=args.headless,
        render_rate=50.0,
        ctrl_rate=1000.0,
        torch_device="cpu",
    )
    robot = HexRobotSimMaverX4(params)
    robot.start()
    robot.step()
    print("[INFO]: Setup complete, starting X4 MIT test.", flush=True)

    dof = 8
    jnt_vel = np.zeros(dof)
    jnt_vel[[0, 2, 4, 6]] = 3.0   # canonical order: drive = joint_wheel1..4

    count = 0
    while robot.is_working():
        robot.set_chs_vel_cmd({
            "vx": 0.0,
            "vy": 1.0,
            "omega": 0.0,
        })
        
        # robot.set_chs_mit_cmd({
        #     "jnt_pos": np.zeros(dof),
        #     "jnt_vel": jnt_vel,
        #     "mit_tau": np.zeros(dof),
        #     "mit_kp": np.zeros(dof),
        #     "mit_kd": np.full(dof, 3.0),
        # })
        robot.step()
        if count % 50 == 0:
            st = robot.get_chassis_state()
            if st is not None:
                cs = st.chs_state
                print(f"[x4] {count}: jnt_len={len(cs.jnt.position)} \n"
                      f"chassis joint eff {cs.jnt.effort} \n"
                      f"chassis joint vel {cs.jnt.velocity} \n"
                      f"chassis joint pos {cs.jnt.position} \n"
                      f"chassis linear {cs.odom.twist.linear} \n"
                      f"chassis angular {cs.odom.twist.angular} \n"
                      , flush=True)
        count += 1
        time.sleep(1.0 / params.ctrl_rate)

    robot.stop()
    print(f"[INFO]: Done ({count} steps).")


if __name__ == "__main__":
    main()
