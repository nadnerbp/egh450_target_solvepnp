#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import TransformStamped

class PoseEstimator:
    def __init__(self):
        rospy.loginfo("Pose Estimator initialising...")

        # Subscribers
        self.image_sub = rospy.Subscriber(
            "/depthai_node/image/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.caminfo_sub = rospy.Subscriber(
            "/depthai_node/camera/camera_info",
            CameraInfo,
            self.caminfo_callback,
            queue_size=1,
        )

        # Publisher
        self.pose_pub = rospy.Publisher(
            "/camera/pose", TransformStamped, queue_size=10
        )

        # TF broadcaster
        self.tfbr = tf2_ros.TransformBroadcaster()

        # State
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        rospy.loginfo("Pose Estimator ready.")

    def caminfo_callback(self, msg):
        """Get intrinsics from CameraInfo."""
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)
        rospy.loginfo_once("Camera intrinsics received.")

    def image_callback(self, msg):
        """Process image and estimate marker pose."""
        if self.camera_matrix is None:
            rospy.logwarn_throttle(5.0, "No camera intrinsics yet.")
            return

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            return

        for i, marker_id in enumerate(ids.flatten()):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.15, self.camera_matrix, self.dist_coeffs
            )

            # Build TransformStamped for this marker
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "camera"

            # child_frame_id is the marker ID (as string)
            t.child_frame_id = str(marker_id)

            # Fill translation
            t.transform.translation.x = float(tvec[0][0][0])
            t.transform.translation.y = float(tvec[0][0][1])
            t.transform.translation.z = float(tvec[0][0][2])

            # Convert rotation vector to quaternion
            rot_mat, _ = cv2.Rodrigues(rvec[0][0])
            qw = np.sqrt(1.0 + rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]) / 2.0
            qx = (rot_mat[2, 1] - rot_mat[1, 2]) / (4.0 * qw)
            qy = (rot_mat[0, 2] - rot_mat[2, 0]) / (4.0 * qw)
            qz = (rot_mat[1, 0] - rot_mat[0, 1]) / (4.0 * qw)

            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw

            # Publish to topic
            self.pose_pub.publish(t)
            # Broadcast TF
            self.tfbr.sendTransform(t)

            rospy.loginfo_throttle(
                2.0,
                f"Published pose for ArUco ID {marker_id} at "
                f"x:{t.transform.translation.x:.2f}, "
                f"y:{t.transform.translation.y:.2f}, "
                f"z:{t.transform.translation.z:.2f}",
            )

def main():
    rospy.init_node("pose_estimator", anonymous=True)
    PoseEstimator()
    rospy.spin()

if __name__ == "__main__":
    main()
