# Autonomous Exploration Pipeline Workspace

This workspace contains the complete robot pipeline for autonomous exploration in Mars-like terrain. It features robust mapping, high-speed navigation, intelligent exploration, and safety boundaries using ArUco markers.

## Contents

- **RTAB-Map**: 3D mapping with LiDAR and cameras
- **Nav2**: Fast and safe navigation using ROS2 navigation stack
- **Explore Lite**: Custom implementation for autonomous exploration with boundary logic
- **ArUco Localization Node**: Detects ArUco tags to set virtual map boundaries
- **Map Saver Node**: Saves generated maps along with the path taken by the robot
---

## Quickstart Instructions

### 1. Clone This Repository

```

git clone https://github.com/iitbmartian/erc_ws.git
cd erc_ws

```

### 2. Build the Workspace

```

colcon build

```

### 3. Source Your Workspace

```

source install/setup.bash

```

---

## Running the Pipeline

1. **Start RTAB-Map Mapping**
    ```
    ros2 launch rtabmap rtabmap_panther.launch.py
    ```

2. **Start Navigation Stack**
    ```
    ros2 launch navigation robot_nav.launch.py
    ```

3. **Start the Explore Lite Node**
    ```
    ros2 launch explore_lite explore.launch.py
    ```

4. **Run the Python Utility Nodes**  
    In separate terminals (after sourcing each terminal):
    ```
    python3 plotter.py
    python3 aruco_localization.py
    ```

---

## Notes

- Make sure you have all ROS 2 dependencies installed (recommended distro: Jazzy or later).
- Always source `install/setup.bash` in every new terminal before running commands.
- Configure launch files and parameter files as per your robot hardware and environment.
---

## License

[MIT] – Feel free to use, modify, and share.

## Contact

For issues or contribution requests, open a GitHub Issue or Pull Request.
