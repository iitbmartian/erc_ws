import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for headless operation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
 
from nav_msgs.msg import OccupancyGrid,Odometry
from visualization_msgs.msg import MarkerArray, Marker
import threading
import sys
import time
import select
import matplotlib.colors as mcolors
import os

MAP_TOPIC     = '/rtabmap/map'
MARKER_TOPIC  = '/aruco_markers'
ODOM_TOPIC    = '/panther/odometry/filtered'

def occgrid_to_numpy(msg: OccupancyGrid):
    h, w = msg.info.height, msg.info.width
    return np.array(msg.data, dtype=np.int8).reshape((h, w))

def extent_from_info(info):
    x0, y0 = info.origin.position.x, info.origin.position.y
    return [x0, x0 + info.width * info.resolution,
            y0, y0 + info.height * info.resolution]

class PlotMapMarkersNoTF(Node):
    def __init__(self):
        self.i=0
        
        super().__init__('plot_map_markers_no_tf')
         
        self.map_msg = None
        self.xy = []
        self.create_subscription(OccupancyGrid, MAP_TOPIC, self.on_map, 10)
        self.create_subscription(MarkerArray, MARKER_TOPIC, self.on_markers, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.listener_callback, 10)
        self.coords=[] 
        self.lock = threading.Lock()
    def listener_callback(self, msg: Odometry):
        with self.lock:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self.coords.append((x, y))
             
         
    def on_map(self, msg: OccupancyGrid):
        with self.lock:
            self.map_msg = msg
             
            map_info_list=[]
            map_info_list.append(msg.info.origin.position.x)
            map_info_list.append(msg.info.origin.position.y)
            map_info_list.append(msg.info.width)
            map_info_list.append(msg.info.height)
            map_info_list.append(msg.info.resolution)
            with open('map_info.txt', 'w') as f:
                f.write(','.join(map(str, map_info_list)))
                 

    def on_markers(self, arr: MarkerArray):
        with self.lock:
            self.xy = []
            for m in arr.markers:
                if m.type in (Marker.POINTS, Marker.SPHERE_LIST, Marker.LINE_STRIP, Marker.LINE_LIST):
                    for p in m.points:
                        self.xy.append((p.x, p.y))
                else:
                    self.xy.append((m.pose.position.x, m.pose.position.y))
            self.xy = self.xy[-3000:]

    def save_map_image(self, filename="map_markers.png"):
        with self.lock:
            if not self.map_msg:
                print("No map received yet.")
                return
            self.i += 1
            grid = occgrid_to_numpy(self.map_msg)
            extent = extent_from_info(self.map_msg.info)
            teal_grey_rgb = mcolors.to_rgb("#CCCCCC")
            cmap = mcolors.ListedColormap([teal_grey_rgb, "#f0f0f0", 'black'])
            bounds = [-1, 0, 1, 2]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            display_grid = np.copy(grid)
            display_grid[display_grid < 0] = -1
            display_grid[display_grid == 0] = 0
            display_grid[display_grid > 0] = 1
            fig, ax = plt.subplots()
            # Set both figure and axes facecolor
            fig.patch.set_facecolor('#CCCCCC')
            ax.set_facecolor('#CCCCCC')
            im = ax.imshow(display_grid, origin='lower', extent=extent,
                           cmap=cmap, norm=norm, interpolation='nearest')
            
            # Remove x and y axes
            ax.axis('off')

            filename_before="map.png"
            plt.savefig(filename_before, bbox_inches='tight', facecolor=fig.get_facecolor(),dpi=400,)
            print(f"Saved intermediate map image to {filename_before}")
            legend_handles = []
            # Add legend patches for map colors
            legend_handles.append(mpatches.Patch(color='#CCCCCC', label='Unknown'))
            legend_handles.append(mpatches.Patch(color='#f0f0f0', label='Free'))
            legend_handles.append(mpatches.Patch(color='black', label='Occupied'))
            # For aruco markers and path, add proxy artists
            blue_marker = plt.Line2D([0], [0], marker='o', color='w', label='Aruco Markers',
                                     markerfacecolor='blue', markeredgecolor='blue', markersize=8, linestyle='None')
            red_line = plt.Line2D([0], [0], color='red', lw=2, label='Path Taken')
            legend_handles.append(mpatches.Patch(color='green', label='initial_pose'))
            legend_handles.append(red_line)
            legend_handles.append(blue_marker)
            if self.xy:
                xs, ys = zip(*self.xy)
                ax.scatter(xs, ys, s=20, marker='o', c='blue', edgecolors='blue', linewidths=1.5, zorder=10)

            if self.coords:
                print(len(self.coords))
                ox, oy = zip(*self.coords)
                ax.plot(ox, oy, c='red', linewidth=1.5, zorder=5)
                
                # Green circle at the EXACT START of path (first coordinate)
                start_x, start_y = self.coords[0]
                ax.scatter(start_x, start_y, s=10, marker='o', c='green', 
                          edgecolors='green', linewidths=1.5, zorder=12)
                
                # Black arrow at the EXACT END of path (if we have at least 2 points)
                if len(self.coords) >= 2:
                    end_x, end_y = self.coords[-1]
                    prev_x, prev_y = self.coords[-2]
                    
                    # Create a fancy arrow patch
                    arrow = FancyArrowPatch((prev_x, prev_y), (end_x, end_y),
                                           connectionstyle="arc3", 
                                           arrowstyle='->', 
                                           mutation_scale=8,  # Size of arrowhead
                                           color='red',
                                           linewidth=2,
                                           zorder=15)
                    ax.add_patch(arrow)

            # Add the custom legend
            # ax.legend(
            #     handles=legend_handles,
            #     loc='best',             # Auto-choose free space
            #     fontsize='small',       # Smaller text
            #     frameon=True,           # Show box (optional)
            #     borderpad=0.2,          # Padding inside the box
            #     handlelength=1.0,       # Shorter lines
            #     handletextpad=0.3,      # Less space between line and label
            #     labelspacing=0.3,       # Vertical spacing between entries
            #     borderaxespad=0.2       # Space between legend and plot edge
            # )
            
            filename="map1.png"
            plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(),dpi=400)
            plt.close(fig)
            print(f"Saved map image to {filename}")

def key_listener(node):
    print("Press 'm' then Enter to save map image. Press Ctrl+C to exit.")
    while rclpy.ok():
        # Use select for non-blocking input
        if select.select([sys.stdin], [], [], 0.1)[0]:
            key = sys.stdin.readline().strip()
            if key.lower() == 'm':
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"map_markers_{timestamp}.png"
                node.save_map_image(filename)

def main():
    rclpy.init()
    node = PlotMapMarkersNoTF()
    listener_thread = threading.Thread(target=key_listener, args=(node,), daemon=True)
    listener_thread.start()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()