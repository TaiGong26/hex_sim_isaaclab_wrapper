# hex_sim_isaaclab_wrapper

HEXFELLOW 机器人仿真封装包 — V1 基于 Isaac Lab。

## 设计

```
用户 → HexRobotSimArcherY6(params)
         ├── set_arm_mit_cmd(dict)
         ├── set_arm_pos_cmd(dict)
         ├── set_arm_pose_cmd(dict)   ← Isaac Lab IKController
         ├── get_arm_state()          → HexDcRoboArmStateStamped(同真实API)
         ├── get_grip_state()         → HexDcRoboGripStateStamped(同真实API)
         └── update()                 ← 同步推进仿真
                  │
         SimInterface(ABC)            ← 仿真器可插拔抽象
                  │
         IsaacLabSimInterface         ← V1 实现
```

## 安装

```bash
# 普通安装
pip install -e pkg_hex/hex_sim_isaaclab_wrapper/

# 带 Isaac Lab 依赖
pip install -e "pkg_hex/hex_sim_isaaclab_wrapper/[isaaclab]"
```

## 运行

```bash
source shells/isaaclab2.1.1_env.sh
python pkg_hex/hex_sim_isaaclab_wrapper/scripts/sim_archer_y6.py --steps 500
```

## V1 范围

- 仅 Archer Y6（手臂 + 可选的 GR100/GP80 夹爪）
- 无底盘机器人
- 无硬件专用方法（reboot/standby/battery 等）
