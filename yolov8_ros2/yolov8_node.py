# Basic ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import ReliabilityPolicy, QoSProfile

# Executor and callback imports
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# ROS2 interfaces
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String

# Image msg parser
from cv_bridge import CvBridge

# Vision model
from ultralytics import YOLO

# Others
import numpy as np
import open3d as o3d
import time, json, torch

class Yolov8Node(Node):
    def __init__(self):
        super().__init__("yolov8_node")
        rclpy.logging.set_logger_level('yolov8_node', rclpy.logging.LoggingSeverity.INFO)
        
        ## Declare parameters for node
        self.declare_parameter("model", "yolov8n-seg.pt")
        model = self.get_parameter("model").get_parameter_value().string_value
        
        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value
        
        self.declare_parameter("threshold", 0.6)
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        
        self.declare_parameter("enable_yolo", True)
        self.enable_yolo = self.get_parameter("enable_yolo").get_parameter_value().bool_value
        
        ## other inits
        self.group_1 = MutuallyExclusiveCallbackGroup() # camera subscribers
        self.group_2 = MutuallyExclusiveCallbackGroup() # vision timer
        
        self.cv_bridge = CvBridge()
        self.yolo = YOLO(model)
        self.yolo.fuse()
        self.color_image_msg = None
        self.depth_image_msg = None
        self.camera_intrinsics = None
        self.pred_image_msg = Image()
        
        # Publishers
        self._item_dict_pub = self.create_publisher(String, "/yolo/prediction/item_dict", 10)
        self._pred_pub = self.create_publisher(Image, "/yolo/prediction/image", 10)

        # Subscribers
        self._color_image_sub = self.create_subscription(Image, "/camera/camera/color/image_raw", self.color_image_callback, qos_profile_sensor_data, callback_group=self.group_1)
        self._depth_image_sub = self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", self.depth_image_callback, qos_profile_sensor_data, callback_group=self.group_1)
        self._camera_info_subscriber = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE), callback_group=self.group_1)

        # 一定間隔で繰り返す（第一引数が間隔）
        self._vision_timer = self.create_timer(0.5, self.object_segmentation, callback_group=self.group_2) # 25 hz

    def color_image_callback(self, msg):
        self.color_image_msg = msg
        
    def depth_image_callback(self, msg):
        self.depth_image_msg = msg
    
    def camera_info_callback(self, msg):
        try:
            if self.camera_intrinsics is None:
                self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
                self.camera_intrinsics.set_intrinsics(msg.width,    #msg.width
                                                    msg.height,       #msg.height
                                                    msg.k[0],         #msg.K[0] -> fx
                                                    msg.k[4],         #msg.K[4] -> fy
                                                    msg.k[2],         #msg.K[2] -> cx
                                                    msg.k[5] )        #msg.K[5] -> cy
                self.get_logger().info('Camera intrinsics have been set!')
            
        except Exception as e:
            self.get_logger().error(f'camera_info_callback Error: {e}')

    def object_segmentation(self):
        if self.enable_yolo and self.color_image_msg is not None:
            self.get_logger().debug("Succesfully acquired color image msg")
            
            # カラー画像から物体認識を行う
            cv_color_image = self.cv_bridge.imgmsg_to_cv2(self.color_image_msg, desired_encoding='bgr8')
            results = self.yolo.predict(
                source=cv_color_image,
                show=False,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device
            )
            self.get_logger().debug("Succesfully predicted")
            
            item_dict = {}
            detection_class = results[0].boxes.cls.cpu().numpy()
            detection_conf = results[0].boxes.conf.cpu().numpy()
            
            for i, cls in enumerate(detection_class):
                # Extract image with yolo predictions
                pred_img = results[0].plot()
                self.pred_image_msg = self.cv_bridge.cv2_to_imgmsg(pred_img, encoding='passthrough')
                self._pred_pub.publish(self.pred_image_msg)

                if results[0].names[int(cls)] == "person":
                    # ボックスの中央のX座標とY座標を求める
                    bbox = results[0].boxes.xyxy.cpu().numpy()[i]
                    x_center = int((bbox[0] + bbox[2]) / 2)
                    y_center = int((bbox[1] + bbox[3]) / 2)

                    # item_dictに必要なデータを加える
                    item_dict[f'person_{i}'] = {
                        'class': results[0].names[int(cls)],
                        'confidence': round(detection_conf[i].tolist(), 3),
                        'Xpos': round(x_center, 3),
                        'Ypos': round(y_center, 3)
                    }

                    # 深さ情報を受け取ったときのみ、
                    if self.depth_image_msg is not None:
                        # 深さ情報を使える形にする
                        np_depth_image = self.cv_bridge.imgmsg_to_cv2(self.depth_image_msg, desired_encoding='passthrough')
                        depth_image_3d = np.dstack((np_depth_image, np_depth_image, np_depth_image))
                        
                        # Z座標を求めて、item_dictに加える
                        depth_value = np_depth_image[y_center, x_center]
                        item_dict[f'person_{i}']['Zpos'] = depth_value.tolist()
                    else:
                        self.get_logger().info("Depth image is not available.")
                
                # コンソールに出力
                self.get_logger().info(f"Detected {len(item_dict)} person(s) in the image")
                self.get_logger().info(f"Item dictionary: {item_dict}")

            # item_dictをPublishする
            item_dict_msg = String()
            item_dict_msg.data = json.dumps(item_dict)
            self._item_dict_pub.publish(item_dict_msg)
            
            self.get_logger().debug("Item dictionary successfully created and published")
            
    def shutdown_callback(self):
        self.get_logger().warn("Shutting down...")
        
def main(args=None):
    rclpy.init(args=args)

    # Instansiate node class
    vision_node = Yolov8Node()

    # Create executor
    executor = MultiThreadedExecutor()
    executor.add_node(vision_node)

    try:
        # Run executor
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    
    finally:
        # Shutdown executor
        vision_node.shutdown_callback()
        executor.shutdown()

if __name__ == "__main__":
    main()
