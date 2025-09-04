#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class PoseEstimator:
    def __init__(self):
        # Set up the CV Bridge
        self.bridge = CvBridge()

        # Subscribe to both YOLOv5 and ArUco detections
        self.sub_object_pose = rospy.Subscriber("/object_pose", Float32MultiArray, self.callback_obj_pose, queue_size=50)
        self.sub_aruco_pose = rospy.Subscriber("/aruco_detection", Float32MultiArray, self.callback_aruco_pose, queue_size=50)
        self.pub_camera_pose = rospy.Publisher("/camera/pose", TransformStamped, queue_size=50)

        # Initialize variables
        self.model_image = None
        self.object_data = {}  # Stores detection data
        self.published_objects = set()
        self.got_camera_info = False
        self.camera_matrix = None
        self.dist_coeffs = None

        # Flags to prevent re-processing
        self.output_printed = {}

        # Set ROS parameters
        self.param_use_compressed = rospy.get_param("~use_compressed", False)

        # Set up subscribers and publishers
        self.sub_info = rospy.Subscriber("~camera_info", CameraInfo, self.callback_info)
        if self.param_use_compressed:
            self.sub_img = rospy.Subscriber("~image_raw/compressed", CompressedImage, self.callback_img)
            self.pub_overlay = rospy.Publisher("~overlay/image_raw/compressed", CompressedImage, queue_size=1)
        else:
            self.sub_img = rospy.Subscriber("~image_raw", Image, self.callback_img)
            self.pub_overlay = rospy.Publisher("~overlay/image_raw", Image, queue_size=1)

        self.tfbr = tf2_ros.TransformBroadcaster()

    def shutdown(self):
        # Unregister subscribers
        self.sub_info.unregister()
        self.sub_img.unregister()

    # Callback for CameraInfo messages
    def callback_info(self, msg_in):
        self.dist_coeffs = np.array(msg_in.D, dtype=np.float32)
        self.camera_matrix = np.array([
            [msg_in.K[0], msg_in.K[1], msg_in.K[2]],
            [msg_in.K[3], msg_in.K[4], msg_in.K[5]],
            [msg_in.K[6], msg_in.K[7], msg_in.K[8]]],
            dtype=np.float32)
        
        if not self.got_camera_info:
            rospy.loginfo("Got camera info")
            self.got_camera_info = True

    # Callback for YOLOv5 object pose messages
    def callback_obj_pose(self, msg_in):
        obj_array = msg_in.data
        if len(obj_array) < 11:
            rospy.logwarn("Object POSE not transmitted correctly")
            return

        class_id = int(obj_array[0])  # Extract the class ID
        corners = obj_array[1:9]
        marker_length_x = obj_array[9]
        marker_length_y = obj_array[10]

        # Verify if we have a known target
        if class_id == 101:
            target = "bag"
        elif class_id == 102:
            target = "human"
        else:
            rospy.logwarn("Unknown target identification")
            return

        # Store detection data
        self.object_data[class_id] = (corners, marker_length_x, marker_length_y)

        # Initialize the flag for this object ID
        if class_id not in self.output_printed:
            self.output_printed[class_id] = False

    # Callback for ArUco detection messages
    def callback_aruco_pose(self, msg_in):
        obj_array = msg_in.data
        if len(obj_array) != 9:
            rospy.logwarn("ArUco POSE not transmitted correctly")
            return

        marker_id = int(obj_array[0])  # Extract the marker ID
        corners = obj_array[1:9]  # Should be 8 elements

        # Known marker size (adjust to your actual marker size in meters)
        marker_length = 0.2  # For example, 5 cm markers

        # Store detection data
        self.object_data[marker_id] = (corners, marker_length, marker_length)

        # Initialize the flag for this marker ID
        if marker_id not in self.output_printed:
            self.output_printed[marker_id] = False

    # Callback for images
    def callback_img(self, msg_in):
        if not self.got_camera_info:
            rospy.logwarn("Waiting for camera info.")
            return

        # Convert ROS image to OpenCV image
        try:
            if self.param_use_compressed:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_in, "bgr8")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg_in, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # If we have object data, use it to estimate pose
        for class_id, data in self.object_data.items():
            # data is (corners, marker_length_x, marker_length_y)
            corners = data[0]
            marker_length_x = data[1]
            marker_length_y = data[2]

            # Ensure we have exactly 8 corner values
            if len(corners) != 8:
                rospy.logwarn("Incorrect number of corners")
                continue

            self.model_image = np.array([
                (corners[0], corners[1]),  # Top-left
                (corners[2], corners[3]),  # Top-right
                (corners[4], corners[5]),  # Bottom-right
                (corners[6], corners[7])   # Bottom-left
            ], dtype=np.float32)

            # Check that we have at least 4 points
            if self.model_image.shape[0] < 4:
                rospy.logwarn("Not enough points for solvePnP")
                continue

            # Create the model object points
            self.model_object = np.array([
                (-marker_length_x / 2, marker_length_y / 2, 0.0),  # Top-left
                (marker_length_x / 2, marker_length_y / 2, 0.0),   # Top-right
                (marker_length_x / 2, -marker_length_y / 2, 0.0),  # Bottom-right
                (-marker_length_x / 2, -marker_length_y / 2, 0.0)  # Bottom-left
            ], dtype=np.float32)

            # Perform pose estimation using solvePnP
            success, rvec, tvec = cv2.solvePnP(self.model_object, self.model_image, self.camera_matrix, self.dist_coeffs)
            
            # For 3m Survey Altitude
            # offsetx = 0.6257
            # offsety = 0.173

            # For 2m Survey Altitude
            # offsetx = 0.4115
            # offsety = 0.104

            # For 1.5m Survey Altitude
            offsetx = 0.3045
            offsety = 0.07

            if success:
                # Check if the output for this object has already been printed
                if self.output_printed.get(class_id, False):
                    continue  # Skip if already printed

                msg_out = TransformStamped()
                msg_out.header = msg_in.header
                msg_out.child_frame_id = f"{class_id}"
                msg_out.transform.translation.x = tvec[0][0] + offsetx
                msg_out.transform.translation.y = tvec[1][0] + offsety
                msg_out.transform.translation.z = tvec[2][0]
                
                # Convert rotation vector to quaternion
                q = tf_conversions.transformations.quaternion_from_euler(rvec[0][0], rvec[1][0], rvec[2][0])
                msg_out.transform.rotation.x = q[0]
                msg_out.transform.rotation.y = q[1]
                msg_out.transform.rotation.z = q[2]
                msg_out.transform.rotation.w = q[3]
                
                rospy.loginfo("Sending Initial Coordinates for ID: %d", class_id)
                rospy.loginfo("Translation x: %f",  msg_out.transform.translation.x)
                rospy.loginfo("Translation y: %f",  msg_out.transform.translation.y)
                rospy.loginfo("Translation z: %f",  msg_out.transform.translation.z)
                
                # Broadcast the pose and publish the message
                self.tfbr.sendTransform(msg_out)
                self.pub_camera_pose.publish(msg_out)

                # Add the object to published set
                self.published_objects.add(class_id)

                # Set the flag to True after printing
                self.output_printed[class_id] = True

                # Visualize the corners
                # for point in self.model_image:
                #     cv2.circle(cv_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), 3)

        # Publish overlay image
        # try:
        #     if self.param_use_compressed:
        #         self.pub_overlay.publish(self.bridge.cv2_to_compressed_imgmsg(cv_image, "png"))
        #     else:
        #         self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        # except (CvBridgeError, TypeError) as e:
        #     rospy.logerr(e)

if __name__ == "__main__":
    rospy.init_node("egh450_target_solvepnp")
    pe = PoseEstimator()
    rospy.loginfo("Estimating Target POSE")

	# Loop here until quit
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
		# Shutdown
        rospy.loginfo("Shutting down...")
        pe.shutdown()
        rospy.loginfo("Done!")





# import rospy
# import cv2
# import numpy as np
# import tf2_ros
# import tf_conversions
# from geometry_msgs.msg import TransformStamped
# from std_msgs.msg import Float32MultiArray
# from sensor_msgs.msg import Image, CompressedImage, CameraInfo
# from cv_bridge import CvBridge, CvBridgeError

# class PoseEstimator:
#     def __init__(self):
#         # Set up the CV Bridge
#         self.bridge = CvBridge()
#         self.sub_object_pose = rospy.Subscriber("/object_pose", Float32MultiArray, self.callback_obj_pose, buff_size=4)
#         self.pub_camera_pose = rospy.Publisher("/camera/pose", TransformStamped, queue_size=50)

#         # Initialize variables
#         self.model_image = None
#         self.object_data = {}
#         self.published_objects = set()
#         self.got_camera_info = False
#         self.camera_matrix = None
#         self.dist_coeffs = None

#         # Flag to ensure output is printed only once per object
#         self.output_printed = {}

#         # Set ROS parameters
#         self.param_use_compressed = rospy.get_param("~use_compressed", False)

#         # Set up subscribers and publishers
#         self.sub_info = rospy.Subscriber("~camera_info", CameraInfo, self.callback_info)
#         if self.param_use_compressed:
#             self.sub_img = rospy.Subscriber("~image_raw/compressed", CompressedImage, self.callback_img)
#             self.pub_overlay = rospy.Publisher("~overlay/image_raw/compressed", CompressedImage, queue_size=1)
#         else:
#             self.sub_img = rospy.Subscriber("~image_raw", Image, self.callback_img)
#             self.pub_overlay = rospy.Publisher("~overlay/image_raw", Image, queue_size=1)

#         self.tfbr = tf2_ros.TransformBroadcaster()

#     def shutdown(self):
#         # Unregister anything that needs it here
#         self.sub_info.unregister()
#         self.sub_img.unregister()

#     # Callback for CameraInfo messages
#     def callback_info(self, msg_in):
#         self.dist_coeffs = np.array([[msg_in.D[0], msg_in.D[1], msg_in.D[2], msg_in.D[3], msg_in.D[4]]], dtype="double")
#         self.camera_matrix = np.array([
#             [msg_in.P[0], msg_in.P[1], msg_in.P[2]],
#             [msg_in.P[4], msg_in.P[5], msg_in.P[6]],
#             [msg_in.P[8], msg_in.P[9], msg_in.P[10]]],
#             dtype="double")
        
#         if not self.got_camera_info:
#             rospy.loginfo("Got camera info")
#             self.got_camera_info = True

#     # Callback for object pose messages
#     def callback_obj_pose(self, msg_in):
#         obj_array = msg_in.data
#         class_id = int(obj_array[0])  # Extract the class ID
#         corners = obj_array[1:9]
#         marker_length_x = obj_array[9]
#         marker_length_y = obj_array[10]

#         # Verify if we have a known target
#         if class_id == 101:
#             target = "Drone"
#         elif class_id == 102:
#             target = "Phone"
#         else:
#             rospy.logwarn("Unknown target identifier")
#             return

#         # Store raw corner data keyed by marker ID
#         self.object_data[class_id] = corners

#         # Initialize the flag for this object ID
#         if class_id not in self.output_printed:
#             self.output_printed[class_id] = False

#         # Create a model object based on detected marker size
#         self.model_object = np.array([
#             (-marker_length_x / 2, marker_length_y / 2, 0.0),  # Top-left
#             (marker_length_x / 2, marker_length_y / 2, 0.0),   # Top-right
#             (marker_length_x / 2, -marker_length_y / 2, 0.0),  # Bottom-right
#             (-marker_length_x / 2, -marker_length_y / 2, 0.0)  # Bottom-left
#         ])

#     # Callback for images
#     def callback_img(self, msg_in):
#         if not self.got_camera_info:
#             rospy.logwarn("Waiting for camera info.")
#             return

#         # Convert ROS image to OpenCV image
#         try:
#             if self.param_use_compressed:
#                 cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_in, "bgr8")
#             else:
#                 cv_image = self.bridge.imgmsg_to_cv2(msg_in, "bgr8")
#         except CvBridgeError as e:
#             rospy.logerr(e)
#             return

#         # If we have object data, use it to estimate pose
#         for class_id, corners in self.object_data.items():
#             # Ensure we have exactly 4 points
#             if len(corners) != 8:
#                 rospy.logwarn("Incorrect number of corners")
#                 continue

#             self.model_image = np.array([
#                 (corners[0], corners[1]),  # Top-left
#                 (corners[2], corners[3]),  # Top-right
#                 (corners[4], corners[5]),  # Bottom-right
#                 (corners[6], corners[7])   # Bottom-left
#             ], dtype=np.float32)

#             # Check that we have at least 4 points
#             if self.model_object.shape[0] < 4 or self.model_image.shape[0] < 4:
#                 rospy.logwarn("Not enough points for solvePnP")
#                 continue

#             # Perform pose estimation using solvePnP
#             success, rvec, tvec = cv2.solvePnP(self.model_object, self.model_image, self.camera_matrix, self.dist_coeffs)

#             if success:
#                 # Check if the output for this object has already been printed
#                 if self.output_printed.get(class_id, False):
#                     continue  # Skip if already printed

#                 msg_out = TransformStamped()
#                 msg_out.header = msg_in.header
#                 msg_out.child_frame_id = f"{class_id}"
#                 msg_out.transform.translation.x = tvec[0]
#                 msg_out.transform.translation.y = tvec[1]
#                 msg_out.transform.translation.z = tvec[2]
                
#                 # Convert rotation vector to quaternion
#                 q = tf_conversions.transformations.quaternion_from_euler(rvec[0], rvec[1], rvec[2])
#                 msg_out.transform.rotation.x = q[0]
#                 msg_out.transform.rotation.y = q[1]
#                 msg_out.transform.rotation.z = q[2]
#                 msg_out.transform.rotation.w = q[3]
                
#                 rospy.loginfo("Translation x: %f",  msg_out.transform.translation.x)
#                 rospy.loginfo("Translation y: %f",  msg_out.transform.translation.y)
#                 rospy.loginfo("Translation z: %f",  msg_out.transform.translation.z)
                
#                 # Broadcast the pose and publish the message
#                 self.tfbr.sendTransform(msg_out)
#                 self.pub_camera_pose.publish(msg_out)

#                 # Add the object to published set
#                 self.published_objects.add(class_id)

#                 # Set the flag to True after printing
#                 self.output_printed[class_id] = True

#                 # Visualize the corners
#                 for point in self.model_image:
#                     cv2.circle(cv_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), 3)

#         # Publish overlay image
#         try:
#             if self.param_use_compressed:
#                 self.pub_overlay.publish(self.bridge.cv2_to_compressed_imgmsg(cv_image, "png"))
#             else:
#                 self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
#         except (CvBridgeError, TypeError) as e:
#             rospy.logerr(e)

# if __name__ == "__main__":
#     rospy.init_node("pose_estimator")
#     pe = PoseEstimator()
#     rospy.spin()


#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class PoseEstimator:
    def __init__(self):
        # Set up the CV Bridge
        self.bridge = CvBridge()

        # Subscribe to both YOLOv5 and ArUco detections
        self.sub_object_pose = rospy.Subscriber("/object_pose", Float32MultiArray, self.callback_obj_pose, queue_size=50)
        self.sub_aruco_pose = rospy.Subscriber("/aruco_detection", Float32MultiArray, self.callback_aruco_pose, queue_size=50)
        self.pub_camera_pose = rospy.Publisher("/camera/pose", TransformStamped, queue_size=50)

        # Initialize variables
        self.model_image = None
        self.object_data = {}  # Stores detection data
        self.published_objects = set()
        self.got_camera_info = False
        self.camera_matrix = None
        self.dist_coeffs = None

        # Flags to prevent re-processing
        self.output_printed = {}

        # Set ROS parameters
        self.param_use_compressed = rospy.get_param("~use_compressed", False)

        # Set up subscribers and publishers
        self.sub_info = rospy.Subscriber("~camera_info", CameraInfo, self.callback_info)
        if self.param_use_compressed:
            self.sub_img = rospy.Subscriber("~image_raw/compressed", CompressedImage, self.callback_img)
            self.pub_overlay = rospy.Publisher("~overlay/image_raw/compressed", CompressedImage, queue_size=1)
        else:
            self.sub_img = rospy.Subscriber("~image_raw", Image, self.callback_img)
            self.pub_overlay = rospy.Publisher("~overlay/image_raw", Image, queue_size=1)

        self.tfbr = tf2_ros.TransformBroadcaster()

    def shutdown(self):
        # Unregister subscribers
        self.sub_info.unregister()
        self.sub_img.unregister()

    # Callback for CameraInfo messages
    def callback_info(self, msg_in):
        self.dist_coeffs = np.array(msg_in.D, dtype=np.float32)
        self.camera_matrix = np.array([
            [msg_in.K[0], msg_in.K[1], msg_in.K[2]],
            [msg_in.K[3], msg_in.K[4], msg_in.K[5]],
            [msg_in.K[6], msg_in.K[7], msg_in.K[8]]],
            dtype=np.float32)
        
        if not self.got_camera_info:
            rospy.loginfo("Got camera info")
            self.got_camera_info = True

    # Callback for YOLOv5 object pose messages
    def callback_obj_pose(self, msg_in):
        obj_array = msg_in.data
        if len(obj_array) < 11:
            rospy.logwarn("Object POSE not transmitted correctly")
            return

        class_id = int(obj_array[0])  # Extract the class ID
        corners = obj_array[1:9]
        marker_length_x = obj_array[9]
        marker_length_y = obj_array[10]

        # Verify if we have a known target
        if class_id == 101:
            target = "Drone"
        elif class_id == 102:
            target = "Phone"
        else:
            rospy.logwarn("Unknown target identification")
            return

        # Store detection data
        self.object_data[class_id] = (corners, marker_length_x, marker_length_y)

        # Initialize the flag for this object ID
        if class_id not in self.output_printed:
            self.output_printed[class_id] = False

    # Callback for ArUco detection messages
    def callback_aruco_pose(self, msg_in):
        obj_array = msg_in.data
        if len(obj_array) != 9:
            rospy.logwarn("ArUco POSE not transmitted correctly")
            return

        marker_id = int(obj_array[0])  # Extract the marker ID
        corners = obj_array[1:9]  # Should be 8 elements

        # Known marker size (adjust to your actual marker size in meters)
        marker_length = 0.2  # For example, 5 cm markers

        # Store detection data
        self.object_data[marker_id] = (corners, marker_length, marker_length)

        # Initialize the flag for this marker ID
        if marker_id not in self.output_printed:
            self.output_printed[marker_id] = False

    # Callback for images
    def callback_img(self, msg_in):
        if not self.got_camera_info:
            rospy.logwarn("Waiting for camera info.")
            return

        # Convert ROS image to OpenCV image
        try:
            if self.param_use_compressed:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_in, "bgr8")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg_in, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # If we have object data, use it to estimate pose
        for class_id, data in self.object_data.items():
            # data is (corners, marker_length_x, marker_length_y)
            corners = data[0]
            marker_length_x = data[1]
            marker_length_y = data[2]

            # Ensure we have exactly 8 corner values
            if len(corners) != 8:
                rospy.logwarn("Incorrect number of corners")
                continue

            self.model_image = np.array([
                (corners[0], corners[1]),  # Top-left
                (corners[2], corners[3]),  # Top-right
                (corners[4], corners[5]),  # Bottom-right
                (corners[6], corners[7])   # Bottom-left
            ], dtype=np.float32)

            # Check that we have at least 4 points
            if self.model_image.shape[0] < 4:
                rospy.logwarn("Not enough points for solvePnP")
                continue

            # Create the model object points
            self.model_object = np.array([
                (-marker_length_x / 2, marker_length_y / 2, 0.0),  # Top-left
                (marker_length_x / 2, marker_length_y / 2, 0.0),   # Top-right
                (marker_length_x / 2, -marker_length_y / 2, 0.0),  # Bottom-right
                (-marker_length_x / 2, -marker_length_y / 2, 0.0)  # Bottom-left
            ], dtype=np.float32)

            # Perform pose estimation using solvePnP
            success, rvec, tvec = cv2.solvePnP(self.model_object, self.model_image, self.camera_matrix, self.dist_coeffs)

            if success:
                # Check if the output for this object has already been printed
                if self.output_printed.get(class_id, False):
                    continue  # Skip if already printed

                msg_out = TransformStamped()
                msg_out.header = msg_in.header
                msg_out.child_frame_id = f"{class_id}"
                msg_out.transform.translation.x = tvec[0][0]
                msg_out.transform.translation.y = tvec[1][0]
                msg_out.transform.translation.z = tvec[2][0]
                
                # Convert rotation vector to quaternion
                q = tf_conversions.transformations.quaternion_from_euler(rvec[0][0], rvec[1][0], rvec[2][0])
                msg_out.transform.rotation.x = q[0]
                msg_out.transform.rotation.y = q[1]
                msg_out.transform.rotation.z = q[2]
                msg_out.transform.rotation.w = q[3]
                
                rospy.loginfo("Translation x: %f",  msg_out.transform.translation.x)
                rospy.loginfo("Translation y: %f",  msg_out.transform.translation.y)
                rospy.loginfo("Translation z: %f",  msg_out.transform.translation.z)
                
                # Broadcast the pose and publish the message
                self.tfbr.sendTransform(msg_out)
                self.pub_camera_pose.publish(msg_out)

                # Add the object to published set
                self.published_objects.add(class_id)

                # Set the flag to True after printing
                self.output_printed[class_id] = True

                # Visualize the corners
                for point in self.model_image:
                    cv2.circle(cv_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), 3)

        # Publish overlay image
        try:
            if self.param_use_compressed:
                self.pub_overlay.publish(self.bridge.cv2_to_compressed_imgmsg(cv_image, "png"))
            else:
                self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except (CvBridgeError, TypeError) as e:
            rospy.logerr(e)

if __name__ == "__main__":
    rospy.init_node("egh450_target_solvepnp")
    pe = PoseEstimator()
    rospy.spin()
