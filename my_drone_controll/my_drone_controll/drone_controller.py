import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Empty
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math
import time

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')
        
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_publisher = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.land_publisher = self.create_publisher(Empty, '/simple_drone/land', 10)
        self.opencv_image_pub_m1 = self.create_publisher(Image, '/openCV/image/m1', 10)
        self.opencv_image_pub_m2 = self.create_publisher(Image, '/openCV/image/m2', 10)
        
        # Subscribers
        self.create_subscription(Image, '/simple_drone/front/image_raw', self.front_camera_callback, 10)
        self.create_subscription(Image, '/simple_drone/bottom/image_raw', self.bottom_camera_callback, 10)
        self.create_subscription(Range, '/simple_drone/sonar/out', self.sonar_callback, 10)
        self.create_subscription(Pose, '/simple_drone/gt_pose', self.gt_pose_callback, 10)
        
        # Timers
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Data holders
        self.front_camera_image = None
        self.bottom_camera_image = None
        self.sonar_data = None
        self.current_position = None
        self.current_yaw = None
        
        # Waypoints
        self.waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_threshold = 1.0  # meters
        self.desired_altitude = 3.0  # Desired altitude in meters
        
        self.temporary_waypoint = None
        self.obstacle_distance_threshold = 2.0  # meters
        self.obstacle_avoidance_distance = 5.0  # meters
        self.obstacle_height_threshold = 5.0  # meters
        self.buffer_distance = 3.0  # Safe buffer distance for obstacle avoidance
        
        self.bridge = CvBridge()
        
        self.get_logger().info('Drone controller node has been started.')
        
        # Send takeoff message and wait for stabilization
        self.takeoff()
        time.sleep(2)  # Short delay to ensure drone is airborne

    def takeoff(self):
        self.get_logger().info('Sending takeoff message.')
        msg = Empty()
        self.takeoff_publisher.publish(msg)

    def land(self):
        self.get_logger().info('Sending land message.')
        msg = Empty()
        self.land_publisher.publish(msg)
    
    def front_camera_callback(self, msg):
        self.front_camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
    def bottom_camera_callback(self, msg):
        self.bottom_camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
    def sonar_callback(self, msg):
        self.sonar_data = msg.range
        
    def gt_pose_callback(self, msg):
        self.current_position = (msg.position.x, msg.position.y)
        orientation_q = msg.orientation
        _, _, yaw = self.euler_from_quaternion(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        self.current_yaw = yaw
        self.get_logger().info(f'Current Position: {self.current_position}, Current Yaw: {math.degrees(self.current_yaw)} degrees')
    
    def timer_callback(self):
        if self.current_position is None or self.current_yaw is None:
            self.get_logger().warn("Current position or yaw is not set yet.")
            return
        
        self.update_trip()
        self.detect_obstacles()
        self.detect_landing_target()
        self.control_drone()
    
    def plan_trip(self, waypoints):
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.get_logger().info(f'Trip planned with {len(waypoints)} waypoints.')

    def get_next_waypoint(self):
        if self.temporary_waypoint:
            return self.temporary_waypoint
        elif self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        else:
            return None
    
    def update_trip(self):
        next_waypoint = self.get_next_waypoint()
        if next_waypoint and self.current_position:
            current_x, current_y = self.current_position
            target_x, target_y = next_waypoint
            distance = math.hypot(target_x - current_x, target_y - current_y)
            self.get_logger().info(f'Distance to next waypoint: {distance} meters.')
            if distance < self.waypoint_threshold:
                if self.temporary_waypoint:
                    self.temporary_waypoint = None
                    self.get_logger().info('Temporary waypoint reached, resuming main path.')
                else:
                    self.current_waypoint_index += 1
                    self.get_logger().info(f'Moving to waypoint {self.current_waypoint_index}')
                    if self.current_waypoint_index == len(self.waypoints):
                        self.get_logger().info('Final waypoint reached, initiating landing.')
                        self.land()
    
    def control_drone(self):
        next_waypoint = self.get_next_waypoint()
        if next_waypoint and self.current_position:
            current_x, current_y = self.current_position
            target_x, target_y = next_waypoint
            bearing = self.calculate_bearing((current_x, current_y), (target_x, target_y))
            self.move_towards_bearing(bearing, current_x, current_y, target_x, target_y)
    
    def calculate_bearing(self, start, end):
        start_x, start_y = start
        end_x, end_y = end
        angle = math.atan2(end_y - start_y, end_x - start_x)
        return math.degrees(angle)
    
    def normalize_angle(self, angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def move_towards_bearing(self, bearing, current_x, current_y, target_x, target_y):
        yaw_error = self.normalize_angle(bearing - math.degrees(self.current_yaw))
        distance = math.hypot(target_x - current_x, target_y - current_y)
        
        # Adjust speed based on distance
        if distance > 5.0:
            speed = 10.0
        elif distance > 2.0:
            speed = 4.5
        else:
            speed = 0.5

        yaw_control = yaw_error * 0.1  # Proportional control for yaw adjustment
        
        self.get_logger().info(f'Bearing: {bearing}, Current Yaw: {math.degrees(self.current_yaw)}, Yaw Error: {yaw_error}, Yaw Control: {yaw_control}, Speed: {speed}')
        
        msg = Twist()
        
        # Stop and turn if the yaw error is too large
        if abs(yaw_error) > 30:
            msg.linear.x = 0.0
        else:
            msg.linear.x = speed
        
        msg.angular.z = np.deg2rad(yaw_control)  # Adjust direction based on yaw control
        
        # Adjust altitude to maintain desired altitude
        if self.sonar_data is not None:
            altitude_error = float(self.desired_altitude) - float(self.sonar_data)
            msg.linear.z = altitude_error * 0.3  # Increased proportional control for altitude
            self.get_logger().info(f'Altitude Error: {altitude_error}, Altitude Control: {msg.linear.z}')
        
        self.cmd_vel_publisher.publish(msg)
    
    def detect_obstacles(self):
        obstacle_detected = False
        obstacle_height = None
        left_edge_detected = False
        right_edge_detected = False
        left_obstacle = None
        right_obstacle = None

        if self.front_camera_image is not None:
            # Convert image to HSV
            hsv_image = cv2.cvtColor(self.front_camera_image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges to exclude (blue sky and green grass)
            lower_blue = np.array([90, 50, 50], dtype=np.uint8)
            upper_blue = np.array([130, 255, 255], dtype=np.uint8)
            lower_green = np.array([35, 50, 50], dtype=np.uint8)
            upper_green = np.array([85, 255, 255], dtype=np.uint8)
            
            # Create masks for blue sky and green grass
            mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
            mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
            
            # Combine masks
            mask = cv2.bitwise_or(mask_blue, mask_green)
            
            # Invert mask to get obstacles
            mask_inv = cv2.bitwise_not(mask)
            masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_inv)
            
            # Convert masked image to grayscale
            gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            
            # Apply GaussianBlur to reduce noise and improve edge detection
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            
            # Use adaptive thresholding for better results in different lighting conditions
            thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Use Canny edge detection
            edges = cv2.Canny(thresh, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Assuming the obstacles are the largest contours detected
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    obstacle_height = cy
                    self.get_logger().info(f'Obstacle detected at ({cx}, {cy}) in the image.')

                    # Visualize the obstacle detection
                    cv2.drawContours(self.front_camera_image, [largest_contour], -1, (0, 255, 0), 3)
                    cv2.circle(self.front_camera_image, (cx, cy), 5, (0, 0, 255), -1)

                    # Check if the obstacle is within 3/4 of the image height
                    image_height = self.front_camera_image.shape[0]
                    if cy < 3 * image_height / 4:
                        self.get_logger().info('Obstacle height is less than 3/4 of the image, planning to go over it.')
                        self.adjust_altitude()
                        self.publish_opencv_image(self.front_camera_image, 'm1')
                        return

                    # Check for edge detection
                    if cx < 320 and self.is_edge_visible(largest_contour, 'left'):
                        left_edge_detected = True
                        left_obstacle = (cx, cy)
                    elif cx > 320 and self.is_edge_visible(largest_contour, 'right'):
                        right_edge_detected = True
                        right_obstacle = (cx, cy)

                    if left_edge_detected and right_edge_detected:
                        # Navigate between the obstacles
                        self.navigate_between_obstacles(left_obstacle, right_obstacle)
                        obstacle_detected = True
                    elif left_edge_detected:
                        # Navigate around the left edge
                        avoidance_bearing = self.normalize_angle(math.degrees(self.current_yaw) + 90)
                        self.avoid_obstacle(avoidance_bearing, 'left')
                        obstacle_detected = True
                    elif right_edge_detected:
                        # Navigate around the right edge
                        avoidance_bearing = self.normalize_angle(math.degrees(self.current_yaw) - 90)
                        self.avoid_obstacle(avoidance_bearing, 'right')
                        obstacle_detected = True

        if obstacle_detected:
            self.get_logger().info('Obstacle detected! Adjusting path.')
            self.publish_opencv_image(self.front_camera_image, 'm1')
        elif self.temporary_waypoint:
            self.get_logger().info('No obstacles, resuming original path.')
            self.temporary_waypoint = None
    
    def adjust_altitude(self):
        msg = Twist()
        msg.linear.z = 1.0  # Increase altitude to fly over obstacle
        self.cmd_vel_publisher.publish(msg)

    def is_edge_visible(self, contour, side):
        # Check if the edge of the obstacle is visible
        if side == 'left':
            for point in contour:
                if point[0][0] < 20:  # Near the left edge of the image
                    return True
        elif side == 'right':
            for point in contour:
                if point[0][0] > 620:  # Near the right edge of the image
                    return True
        return False

    def avoid_obstacle(self, bearing, side):
        current_x, current_y = self.current_position
        temp_x, temp_y = self.calculate_new_position((current_x, current_y), bearing, self.obstacle_avoidance_distance + self.buffer_distance)
        self.temporary_waypoint = (temp_x, temp_y)
        self.get_logger().info(f'Setting temporary waypoint to avoid {side} edge obstacle: {self.temporary_waypoint}')
        self.publish_opencv_image(self.front_camera_image, 'm1')

    def navigate_between_obstacles(self, left_obstacle, right_obstacle):
        left_cx, left_cy = left_obstacle
        right_cx, right_cy = right_obstacle

        # Calculate the middle point between the two obstacles
        middle_cx = (left_cx + right_cx) / 2
        middle_cy = (left_cy + right_cy) / 2

        # Determine the bearing to navigate through the middle
        middle_bearing = self.calculate_bearing(self.current_position, (middle_cx, middle_cy))
        temp_x, temp_y = self.calculate_new_position(self.current_position, middle_bearing, self.obstacle_avoidance_distance + self.buffer_distance)
        self.temporary_waypoint = (temp_x, temp_y)
        self.get_logger().info(f'Setting temporary waypoint to navigate between obstacles: {self.temporary_waypoint}')
        self.publish_opencv_image(self.front_camera_image, 'm1')

    def calculate_new_position(self, start, bearing, distance):
        start_x, start_y = start
        bearing = math.radians(bearing)
        
        new_x = start_x + distance * math.cos(bearing)
        new_y = start_y + distance * math.sin(bearing)
        
        return new_x, new_y
    
    def detect_landing_target(self):
        if self.bottom_camera_image is not None:
            hsv_image = cv2.cvtColor(self.bottom_camera_image, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            mask = cv2.inRange(hsv_image, lower_red, upper_red)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    self.get_logger().info(f'Red circle detected at ({cx}, {cy}).')
                    self.land_on_target(cx, cy)
    
    def land_on_target(self, cx, cy):
        msg = Twist()
        # Simple logic to center the drone over the red circle
        if cx < 320:  # Assuming 640x480 image resolution
            msg.linear.y = 0.2  # Increased speed for centering
        elif cx > 320:
            msg.linear.y = -0.2  # Increased speed for centering
        
        if cy < 240:
            msg.linear.z = -0.2  # Increased speed for centering
        elif cy > 240:
            msg.linear.z = 0.2  # Increased speed for centering
        
        if abs(cx - 320) < 10 and abs(cy - 240) < 10:
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = -0.5  # Land
        
        self.cmd_vel_publisher.publish(msg)
        self.publish_opencv_image(self.bottom_camera_image, 'm2')

    def publish_opencv_image(self, cv_image, topic_suffix):
        try:
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
            if topic_suffix == 'm1':
                self.opencv_image_pub_m1.publish(ros_image)
            elif topic_suffix == 'm2':
                self.opencv_image_pub_m2.publish(ros_image)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z

def main(args=None):
    rclpy.init(args=args)
    drone_controller = DroneController()
    
    # Example waypoints
    waypoints = [
        (-38.37, -145.04),  # Example coordinate
        (64.40, -137.31),  # Example coordinate
        (-5.60, -4.64)   # Example coordinate
    ]
    drone_controller.plan_trip(waypoints)
    
    rclpy.spin(drone_controller)
    drone_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
