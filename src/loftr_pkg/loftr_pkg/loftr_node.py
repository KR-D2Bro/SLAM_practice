import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from interface_pkg.msg import LoFTRMatches
from interface_pkg.srv import LoFTRMatchSrv
from cv_bridge import CvBridge
import cv2
import numpy as np

try:
    import torch
    from kornia.feature import LoFTR
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "LoFTR node requires torch and kornia. Install via "
        "`pip install torch einops yacs kornia`."
    ) from exc


class LoFTRNode(Node):
    """ROS2 node scaffold for running LoFTR matching."""

    def __init__(self) -> None:
        super().__init__("loftr_node")
        self.declare_parameter("pretrained", "outdoor")
        self.declare_parameter("match_srv", "/LoFTR/match")
        self.declare_parameter("kf_ping_topic","/LoFTR/kf_ping")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("visualize", True)

        pretrained = self.get_parameter("pretrained").get_parameter_value().string_value
        device_param = self.get_parameter("device").get_parameter_value().string_value
        requested_device = torch.device(device_param)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn(
                f"CUDA requested ('{device_param}') but not available; falling back to CPU."
            )
            requested_device = torch.device("cpu")
        self.device = requested_device

        self.matcher = LoFTR(pretrained=pretrained)
        self.matcher = self.matcher.to(self.device).eval()
        self.get_logger().info(f"LoFTR initialized with '{pretrained}' weights.")
        self.get_logger().info(f"Using device: {self.device}")

        self.bridge = CvBridge()
        self.visualize = (
            self.get_parameter("visualize").get_parameter_value().bool_value
        )
        self.loftr_srv = self.get_parameter("match_srv").value
        self.kf_ping_topic = self.get_parameter("kf_ping_topic").value

        self.loftr_match_srv = self.create_service(
            LoFTRMatchSrv, self.loftr_srv, self.loftr_match_callback
        )
        self.kf_ping_sub = self.create_subscription(
            Bool, self.kf_ping_topic, self.kf_ping_callback, 10
        )

        self._last_keyFrame = None
        self._last_image0 = None

    def loftr_match_callback(self, request, response):
        if self._last_keyFrame is None:
            self._last_keyFrame = request.image
            response.is_first = 1
            self._last_image0 = request.image
            return response
        
        response.is_first = 0
        response.loftr_matches = self.try_match(self._last_keyFrame, request.image)
        self._last_keyFrame = request.image
        self._last_image0 = request.image
        return response

    def kf_ping_callback(self, msg) -> None:
        self._last_keyFrame = self._last_image0

    def try_match(self, ref_image, cur_image) -> LoFTRMatches:
        if ref_image is None or cur_image is None:
            return

        try:
            last_cv = self.bridge.imgmsg_to_cv2(ref_image, desired_encoding="mono8")
            curr_cv = self.bridge.imgmsg_to_cv2(cur_image, desired_encoding="mono8")

            orig_h, orig_w = last_cv.shape[:2]
            target_w, target_h = 640, 480
            last_small = cv2.resize(last_cv, (target_w, target_h), interpolation=cv2.INTER_AREA)
            curr_small = cv2.resize(curr_cv, (target_w, target_h), interpolation=cv2.INTER_AREA)
            sx = orig_w / target_w
            sy = orig_h / target_h

            last_tensor = self._cv_to_tensor(last_small)
            curr_tensor = self._cv_to_tensor(curr_small)
        except Exception as exc:
            self.get_logger().warn(f"Image conversion failed: {exc}")
            return

        input_dict = {"image0": last_tensor, "image1": curr_tensor}
        with torch.inference_mode():
            output = self.matcher(input_dict)
        mkpts0 = output["keypoints0"].cpu().numpy()
        mkpts1 = output["keypoints1"].cpu().numpy()
        mconf = output["confidence"].cpu().numpy()
        self.get_logger().debug(
            f"Matched {mkpts0.shape[0]} points (device={self.device})."
        )

        mkpts0[:, 0] *= sx; mkpts0[:, 1] *= sy
        mkpts1[:, 0] *= sx; mkpts1[:, 1] *= sy
        # response matches to slamsystem node.
        matches_msg = LoFTRMatches()
        matches_msg.header.stamp = self.get_clock().now().to_msg()
        matches_msg.header.frame_id = "cam0"
        matches_msg.keypoints0_u = mkpts0[:, 0].astype(np.float32).tolist()
        matches_msg.keypoints0_v = mkpts0[:, 1].astype(np.float32).tolist()
        matches_msg.keypoints1_u = mkpts1[:, 0].astype(np.float32).tolist()
        matches_msg.keypoints1_v = mkpts1[:, 1].astype(np.float32).tolist()
        matches_msg.confidence = mconf.astype(np.float32).tolist()

        # visualize result
        if self.visualize:
            self._show_matches(last_cv, curr_cv, mkpts0, mkpts1, mconf)

        return matches_msg

    def _cv_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert grayscale OpenCV image to normalized tensor (1,1,H,W)."""
        tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0)

    def _show_matches(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        pts0: np.ndarray,
        pts1: np.ndarray,
        conf: np.ndarray,
    ) -> None:
        """Simple side-by-side imshow of matches."""
        img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        h = max(img0_color.shape[0], img1_color.shape[0])
        w0 = img0_color.shape[1]
        w1 = img1_color.shape[1]
        canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
        canvas[: img0_color.shape[0], :w0] = img0_color
        canvas[: img1_color.shape[0], w0 : w0 + w1] = img1_color

        # Normalize confidence to [0,1] for color mapping.
        if conf.size > 0:
            conf_min, conf_max = conf.min(), conf.max()
            conf_norm = (conf - conf_min) / (conf_max - conf_min + 1e-6)
        else:
            conf_norm = np.zeros_like(conf)

        for (p0, p1, c) in zip(pts0, pts1, conf_norm):
            c_int = float(c)
            color = (0, int(255 * (1 - c_int)), int(255 * c_int))  # BGR from green→red
            p0_int = (int(p0[0]), int(p0[1]))
            p1_int = (int(p1[0]) + w0, int(p1[1]))
            cv2.circle(canvas, p0_int, 2, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p1_int, 2, color, -1, lineType=cv2.LINE_AA)
            cv2.line(canvas, p0_int, p1_int, color, 1, lineType=cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"matches: {pts0.shape[0]}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("LoFTR Matches", canvas)
        cv2.waitKey(1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LoFTRNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
