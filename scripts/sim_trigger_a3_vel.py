#!/usr/bin/env python3

import argparse
import time

import numpy as np

from hex_sim_isaaclab_wrapper import HexRobotSimTriggerA3, HexRobotSimTriggerA3Params


def _measure(
    robot: HexRobotSimTriggerA3,
    vx: float,
    vy: float,
    omega: float,
    settle: int,
    measure: int,
    rate: float,
) -> np.ndarray:
    """Command a twist for `settle` steps, then average the odom twist.

    Returns:
        Mean `(linear.x, linear.y, angular.z)` over the `measure` samples.
    """
    for _ in range(settle):
        robot.set_chs_vel_cmd(vx, vy, omega)
        robot.step()
        time.sleep(1.0 / rate)
    samples = []
    for _ in range(measure):
        robot.set_chs_vel_cmd(vx, vy, omega)
        robot.step()
        time.sleep(1.0 / rate)
        st = robot.get_chassis_state()
        if st is not None:
            tw = st.chs_state.odom.twist
            samples.append((tw.linear.x, tw.linear.y, tw.angular.z))
    if not samples:
        return np.zeros(3)
    return np.mean(np.array(samples), axis=0)


def main() -> None:
    """Run the Trigger A3 velocity-level ground-speed verification."""
    parser = argparse.ArgumentParser(
        description="Trigger A3 velocity-level omni ground-speed test (Ctrl+C to stop)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--settle", type=int, default=300, help="Settle steps per phase")
    parser.add_argument("--measure", type=int, default=200, help="Sample steps per phase")
    parser.add_argument("--tol", type=float, default=0.15, help="PASS tolerance")
    parser.add_argument("--summary_every", type=int, default=10,
                        help="Print a cumulative summary every N loops")
    args = parser.parse_args()

    params = HexRobotSimTriggerA3Params(
        isaac_headless=args.headless,
        render_rate=100.0,
        ctrl_rate=1000.0,
        torch_device="cpu",
    )
    robot = HexRobotSimTriggerA3(params)
    robot.start()
    print("[INFO]: Setup complete, starting A3 VEL ground-speed test (Ctrl+C to stop).", flush=True)

    # phase name → commanded twist → which axis gates the PASS.
    phases = [
        ("forward", (1.0, 0.0, 0.0), "vx"),
        ("lateral", (0.0, 1.0, 0.0), "vy"),
        ("rotate", (0.0, 0.0, 1.0), "omega"),
        ("stop", (0.0, 0.0, 0.0), "all"),
    ]

    loop = 0
    total_phases = total_passes = 0
    try:
        while True:
            loop += 1
            for name, (vx, vy, omega), gate in phases:
                vx_m, vy_m, om_m = _measure(robot, vx, vy, omega, args.settle, args.measure,
                                            params.ctrl_rate)
                if gate == "vx":
                    ok = abs(vx_m - vx) <= args.tol * abs(vx)
                elif gate == "vy":
                    ok = abs(vy_m - vy) <= args.tol * abs(vy)
                elif gate == "omega":
                    ok = abs(abs(om_m) - abs(omega)) <= args.tol * abs(omega)
                elif gate == "all":
                    ok = abs(vx_m) <= args.tol and abs(vy_m) <= args.tol and abs(om_m) <= args.tol
                else:
                    ok = False
                total_phases += 1
                if ok:
                    total_passes += 1

                sign_note = ""
                if gate == "omega" and om_m < 0.0:
                    sign_note = "  !! SIGN INVERTED (cmd +omega, measured -omega)"
                print(f"[a3 VEL][loop {loop}] {name:8s} cmd=({vx},{vy},{omega}) "
                      f"measured vx={vx_m:+.3f} vy={vy_m:+.3f} omega={om_m:+.3f} "
                      f"-> {'PASS' if ok else 'FAIL'}{sign_note}", flush=True)
            if loop % args.summary_every == 0:
                pct = 100.0 * total_passes / total_phases
                print(f"[SUMMARY] loop {loop}: {total_passes}/{total_phases} phases PASS "
                      f"({pct:.1f}%)", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        robot.stop()
    pct = 100.0 * total_passes / total_phases if total_phases else 0.0
    print(f"[INFO]: Stopped after {loop} loops. Overall {total_passes}/{total_phases} "
          f"phases PASS ({pct:.1f}%).")


if __name__ == "__main__":
    main()
