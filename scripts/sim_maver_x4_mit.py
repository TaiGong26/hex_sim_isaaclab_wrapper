#!/usr/bin/env python3
"""sim_maver_x4_mit.py — Maver X4 per-joint MIT matching test.

For each of the 8 canonical joints individually, sends a velocity command on
that joint alone, then reads back the chassis state to verify that the
commanded index ↔ canonical joint name ↔ actual USD joint all match.

Example:

```bash
source shells/isaaclab2.1.1_env.sh
python pkg_hex/hex_sim_isaaclab_wrapper/scripts/sim_maver_x4_mit.py --headless
```

For each index i ∈ [0, 8):
- sends `jnt_vel = e_i` (index i is 2.0, all others 0), steps several times,
  then reads back the state and asserts the dominant velocity index == i
  with a sufficiently large magnitude.

Expected: each wheel moves only when commanded — drive wheels
`joint_wheel1..4` (canonical indices [0,2,4,6]) and steering joints
`joint_yaw1..4` (canonical indices [1,3,5,7]) each match one-to-one.
"""

import argparse
import time

import numpy as np

from hex_sim_isaaclab_wrapper import HexRobotSimMaverX4, HexRobotSimMaverX4Params
from hex_sim_isaaclab_wrapper.chassis.robot_maver_x4 import JOINT_STATE_NAME

# Steps to hold each single-joint command before reading back.
_STEPS_PER_INDEX = 60
# Commanded velocity magnitude [rad/s].
_CMD_VEL = 2.0
# Min |velocity| [rad/s] for a joint to count as "moved".
_VEL_THRESHOLD = 0.05
# Per-joint MIT kd (steering yaw = 20, drive wheel = 200 — matches the cfg
# actuator damping); kp = 0 so the command is pure velocity tracking.
# Derived by name from `JOINT_STATE_NAME` so a canonical reordering cannot
# silently misalign the kd with the joint type (wheel-first canonical).
_KD = np.where(
    [n.startswith("joint_wheel") for n in JOINT_STATE_NAME], 200.0, 20.0,
).astype(np.float64)


def main() -> int:
    """Run the per-joint MIT matching test and print PASS/FAIL per index.

    Args:
        --headless: Run without a GUI window.

    Returns:
        Exit code: 0 if all joints PASS, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Maver X4 per-joint MIT matching test")
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
    print("[INFO]: Setup complete, starting per-joint matching test.", flush=True)

    dof = len(JOINT_STATE_NAME)
    results = []

    for i in range(dof):
        jnt_vel = np.zeros(dof)
        jnt_vel[i] = _CMD_VEL
        for _ in range(_STEPS_PER_INDEX):
            robot.set_chs_mit_cmd({
                "jnt_pos": np.zeros(dof),
                "jnt_vel": jnt_vel,
                "mit_tau": np.zeros(dof),
                "mit_kp": np.zeros(dof),
                "mit_kd": _KD,
            })
            robot.step()
            time.sleep(1.0 / params.ctrl_rate)

        st = robot.get_chassis_state()
        if st is None:
            results.append((i, "FAIL", "no state"))
            continue
        vel = np.asarray(st.chs_state.jnt.velocity, dtype=np.float32)
        dominant = int(np.argmax(np.abs(vel)))
        moved = np.where(np.abs(vel) > _VEL_THRESHOLD)[0].tolist()
        ok = (dominant == i) and (abs(vel[i]) > _VEL_THRESHOLD)
        results.append((
            i,
            "PASS" if ok else "FAIL",
            f"moved={moved} dominant={dominant} "
            f"vel={np.round(vel, 3).tolist()}",
        ))

    print("\n=== Maver X4 per-joint matching result ===", flush=True)
    all_ok = True
    for i, status, detail in results:
        all_ok &= (status == "PASS")
        print(f"  [{status}] canonical[{i}] = {JOINT_STATE_NAME[i]:12s} {detail}",
              flush=True)
    n_pass = sum(1 for r in results if r[1] == "PASS")
    print(f"\nRESULT: {'ALL PASS' if all_ok else 'FAILURES PRESENT'} "
          f"({n_pass}/{dof})", flush=True)

    robot.stop()
    print("[INFO]: Done.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
