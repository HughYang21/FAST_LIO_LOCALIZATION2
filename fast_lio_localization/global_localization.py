#!/usr/bin/env python3

import copy
# import threading
# import time

import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
# from nav_msgs.msg import Odometry
# from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Header
import numpy as np
import tf2_geometry_msgs
import tf2_ros
import tf_transformations
import ros2_numpy
import logging

logging.getLogger("ros2_numpy").setLevel(logging.ERROR)


class FastLIOLocalization(Node):
    def __init__(self):
        super().__init__("fast_lio_localization")
        self.global_map = None
        self.T_map_to_odom = np.eye(4)
        self.cur_odom = None
        self.cur_scan = None
        self.initialized = False

        self.declare_parameters(
            namespace="",
            parameters=[
                ("map_voxel_size", 0.4),
                ("scan_voxel_size", 0.1),
                ("freq_localization", 0.5),
                ("freq_global_map", 0.25),
                ("localization_threshold", 0.8),
                ("fov", 6.28319),
                ("fov_far", 300),
                ("pcd_map_topic", "/map"),
                ("pcd_map_path", ""),
            ],
        )

        qos_profile_reliable = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.pcd_header = Header()
        self.fast_loop_cbg = MutuallyExclusiveCallbackGroup()
        self.latest_transform = tf2_ros.TransformStamped()
        # Broadcast a temporary identity transform so the 'map' frame exists at startup
        self.latest_transform.header.stamp = self.get_clock().now().to_msg()
        self.latest_transform.header.frame_id = "map"
        self.latest_transform.child_frame_id = "odom"
        
        self.latest_transform.transform.translation.x = 0.0
        self.latest_transform.transform.translation.y = 0.0
        self.latest_transform.transform.translation.z = 0.0
        self.latest_transform.transform.rotation.x = 0.0
        self.latest_transform.transform.rotation.y = 0.0
        self.latest_transform.transform.rotation.z = 0.0
        self.latest_transform.transform.rotation.w = 1.0

        # self.pub_global_map = self.create_publisher(PointCloud2, self.get_parameter("pcd_map_topic").value, 10)
        self.pub_pc_in_map = self.create_publisher(PointCloud2, "/cur_scan_in_map", 10)
        self.pub_submap = self.create_publisher(PointCloud2, "/submap", 10)
        # self.pub_map_to_odom = self.create_publisher(Odometry, "/map_to_odom", 10)
        self.pub_localization_done = self.create_publisher(Bool, "/localization/initiated", qos_profile_reliable)

        self.get_logger().info("Waiting for global map...")
        # global_map_msg = wait_for_message(msg_type = PointCloud2, node = self, topic = "/cloud_pcd")[1]
        # self.initialize_global_map(global_map_msg)
        
        self.initialize_global_map()
        self.get_logger().info("Global map received.")
        
        self.create_subscription(PointCloud2, "/cloud_registered", self.cb_save_cur_scan, 1, callback_group=self.fast_loop_cbg)
        # self.create_subscription(Odometry, "/Odometry", self.cb_save_cur_odom, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.cb_initialize_pose, 1)

        self.timer_localisation = self.create_timer(1.0 / self.get_parameter("freq_localization").value, self.localisation_timer_callback)
        # self.timer_global_map = self.create_timer(1/ self.get_parameter("freq_global_map").value, self.global_map_callback)

    # def global_map_callback(self):
    #     # self.get_logger().info(np.array(self.global_map.points).shape)
    #     header = Header()
    #     header.stamp = self.get_clock().now().to_msg()
    #     header.frame_id = "map"
    #     self.publish_point_cloud(self.pub_global_map, header, np.array(self.global_map.points))
        
    def pose_to_mat(self, pose):
        trans = np.eye(4)
        trans[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        trans[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]
        return trans
    
    def msg_to_array(self, pc_msg):
        pc_array = ros2_numpy.numpify(pc_msg)
        return pc_array["xyz"]
    
    def registration_at_scale(self, scan, map, initial, scale):
        result_icp = o3d.pipelines.registration.registration_icp(
        self.voxel_down_sample(scan, self.get_parameter("scan_voxel_size").value * scale),
        self.voxel_down_sample(map, self.get_parameter("map_voxel_size").value * scale),
        1.0 * scale,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
        )
        return result_icp.transformation, result_icp.fitness
            
    def inverse_se3(self, trans):
        trans_inverse = np.eye(4)
        # R
        trans_inverse[:3, :3] = trans[:3, :3].T
        # t
        trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
        return trans_inverse

    def publish_point_cloud(self, publisher, header, pc):
        data = dict()
        data["xyz"] = pc[:, :3]
        
        # if pc.shape[1] == 3:
        #     data["intensity"] = np.ones((pc.shape[0], 1))
        #     data["rgb"] = np.ones((pc.shape[0], 1))
        # else:
            # data["rgb"] = np.ones_like(pc)
        msg = ros2_numpy.msgify(PointCloud2, data)
        msg.header = header
        if len(msg.fields) == 4:
            msg.point_step = 16
        else:
            msg.point_step = 12
            
        publisher.publish(msg)
        
    def crop_global_map_in_FOV(self, pose_estimation):  # T_map_to_odom
        try:
            # Look up exactly what the odom -> base_link tree looked like 
            # at the exact millisecond this point cloud frame was captured!
            odom_to_sensor_stamped = self.tf_buffer.lookup_transform(
                target_frame="odom",
                source_frame="lidar",
                time=self.pcd_header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # Fallback fallback: if tf cache didn't populate yet, broadcast with 'now' safely
            self.get_logger().warn(f"TF2 lookup fallback active: {str(e)}")
            return None

        trans = odom_to_sensor_stamped.transform.translation
        rot = odom_to_sensor_stamped.transform.rotation

        # Assemble the T_odom_to_sensor 4x4 matrix natively
        T_odom_to_sensor = np.eye(4)
        T_odom_to_sensor[:3, 3] = [trans.x, trans.y, trans.z]
        quat = [rot.x, rot.y, rot.z, rot.w]
        T_odom_to_sensor[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]

        T_map_to_sensor = np.matmul(pose_estimation, T_odom_to_sensor)
        T_sensor_to_map = self.inverse_se3(T_map_to_sensor)

        global_map_in_map = np.array(self.global_map.points)
        global_map_in_map = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
        global_map_in_sensor = np.matmul(T_sensor_to_map, global_map_in_map.T).T

        if self.get_parameter("fov").value > 3.14:
            indices = np.where(
                (global_map_in_sensor[:, 0] < self.get_parameter("fov_far").value)
                & (np.abs(np.arctan2(global_map_in_sensor[:, 1], global_map_in_sensor[:, 0])) < self.get_parameter("fov").value / 2.0)
            )
        else:
            indices = np.where(
                (global_map_in_sensor[:, 0] > 0)
                & (global_map_in_sensor[:, 0] < self.get_parameter("fov_far").value)
                & (np.abs(np.arctan2(global_map_in_sensor[:, 1], global_map_in_sensor[:, 0])) < self.get_parameter("fov").value / 2.0)
            )
        global_map_in_FOV = o3d.geometry.PointCloud()
        global_map_in_FOV.points = o3d.utility.Vector3dVector(np.squeeze(global_map_in_map[indices, :3]))

        header = odom_to_sensor_stamped.header
        header.frame_id = "map"
        self.publish_point_cloud(self.pub_submap, header, np.array(global_map_in_FOV.points)[::10])

        return global_map_in_FOV

    def global_localization(self, pose_estimation):  # T_map_to_odom
        scan_tobe_mapped = copy.copy(self.cur_scan)
        global_map_in_FOV = self.crop_global_map_in_FOV(pose_estimation)
        
        if global_map_in_FOV is not None:
            transformation, _ = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5)
            
            transformation, fitness = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=1)
            
            if fitness > self.get_parameter("localization_threshold").value:
                self.T_map_to_odom = transformation
                self.publish_odom(transformation)
            else:
                self.get_logger().warn(f"Fitness score {fitness} less than localization threshold {self.get_parameter('localization_threshold').value}")

    def voxel_down_sample(self, pcd, voxel_size):
        # print(pcd)
        
        try:
            pcd_down = pcd.voxel_down_sample(voxel_size)
        
        except Exception as e:
            # for opend3d 0.7 or lower
            pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
            
        return pcd_down

    # def cb_save_cur_odom(self, msg):
    #     self.cur_odom = msg
        
    def cb_save_cur_scan(self, msg):
        pc = self.msg_to_array(msg)
        self.pcd_header = msg.header
        self.cur_scan = o3d.geometry.PointCloud()
        self.cur_scan.points = o3d.utility.Vector3dVector(pc)
        self.publish_point_cloud(self.pub_pc_in_map, msg.header, pc)
        self.latest_transform.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.latest_transform)
        # self.get_logger().info(f"Current scan received and published at time {self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9}.")

    def initialize_global_map(self): #, pc_msg):
        # self.global_map = o3d.geometry.PointCloud()
        # self.global_map.points = o3d.utility.Vector3dVector(self.msg_to_array(pc_msg)[:, :3])
        self.global_map = o3d.io.read_point_cloud(self.get_parameter("pcd_map_path").value)
        self.global_map = self.voxel_down_sample(self.global_map, self.get_parameter("map_voxel_size").value)
        # o3d.io.write_point_cloud("/home/wheelchair2/laksh_ws/pcds/lab_map_with_outside_corridor (with ground pcd)_downsampled.pcd", self.global_map)
        self.get_logger().info("Global map received.")

    def cb_initialize_pose(self, msg):  # pose in map frame
        try:
            odom_to_base_transform = self.tf_buffer.lookup_transform(
                'base_link',
                'odom',
                tf2_ros.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().warn(f'Transform failed: {e}')
            return

        T_map_to_base = self.pose_to_mat(msg.pose.pose)
        self.initialized = True
        self.get_logger().info("Initial pose received.")
        self.pub_localization_done.publish(Bool(data=True))

        odom_pos = odom_to_base_transform.transform.translation
        odom_ori = odom_to_base_transform.transform.rotation
        T_base_to_odom = np.eye(4)
        T_base_to_odom[0, 3] = odom_pos.x
        T_base_to_odom[1, 3] = odom_pos.y
        T_base_to_odom[2, 3] = odom_pos.z
        q_odom = [odom_ori.x, odom_ori.y, odom_ori.z, odom_ori.w]
        T_base_to_odom[:3, :3] = tf_transformations.quaternion_matrix(q_odom)[:3, :3]
        
        if self.cur_scan is not None:
            self.global_localization(np.matmul(T_map_to_base, T_base_to_odom))
            
    def publish_odom(self, transform):
        if self.cur_scan is None:
            return
        scan_time = self.pcd_header.stamp

        # broadcast the raw ICP result stamped at the scan_time
        t = tf2_ros.TransformStamped()
        t.header.stamp = scan_time  # Match the past scan history window
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        t.transform.translation.x = transform[0, 3]
        t.transform.translation.y = transform[1, 3]
        t.transform.translation.z = transform[2, 3]

        quat = tf_transformations.quaternion_from_matrix(transform)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)
        self.latest_transform = t  # Store the latest transform for future use

    def localisation_timer_callback(self):
        if not self.initialized:
            self.get_logger().info("Waiting for initial pose... Broadcasting default map->odom.")
            try:
                self.latest_transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    'odom',
                    tf2_ros.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
            except Exception as e:
                self.get_logger().warn(f'Transform failed: {e}')
                return
            self.latest_transform.header.frame_id = "map"
            self.latest_transform.child_frame_id = "odom"          
            self.tf_broadcaster.sendTransform(self.latest_transform)
            return
        
        if self.cur_scan is not None:
            self.global_localization(self.T_map_to_odom)


def main(args=None):
    rclpy.init(args=args)
    node = FastLIOLocalization()
    try:
        mt_executor = MultiThreadedExecutor()
        mt_executor.add_node(node) 
        while rclpy.ok():
            mt_executor.spin_once(timeout_sec=10.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()