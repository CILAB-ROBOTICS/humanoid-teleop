import cv2
import numpy as np
from tactile_collecting.sensors.sensors import SensorEnv  # ★ 너의 경로에 맞게 수정 필요


def visualize_tactile(ports, stack_num=1, adaptive_calibration=False, normalize=True):
    """
    실시간 tactile visualization 함수
    """
    stage_dummy = DummyStage()
    env = SensorEnv(
        ports=ports,
        stack_num=stack_num,
        adaptive_calibration=adaptive_calibration,
        stage=stage_dummy,
        normalize=normalize
    )

    # print("📡 Tactile Visualization Started!")
    # print("Press 'q' to exit.")
    sensitivity = 40  # 초기값
    env.set_resistance(sensitivity)

    try:
        while True:
            images = env.get()             # shape: [stack_num, H, W]
            fps = env.fps

            if isinstance(images, list):
                images = np.array(images)


            # stack_num=1 → 단일 이미지
            # stack_num>1 → 시간 스택이미지를 가로로 concat
            if images.ndim == 3:
                vis_img = np.concatenate(images, axis=1)
            else:
                vis_img = images[0]

            # print(vis_img.min(), vis_img.max())

            # Normalize for visualization
            vis_img = vis_img.astype(np.float32)

            abs_min = 0
            abs_max = 3

            vis_img = (vis_img - abs_min) / (abs_max - abs_min)
            vis_img = np.clip(vis_img, 0, 1)
            vis_img = (vis_img * 255).astype(np.uint8)

            # 컬러맵 적용
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_INFERNO)




            # ===========================================
            # 🔥🔥  여기서 원하는 크기로 조절  🔥🔥
            # ===========================================
            target_width = 512  # 너가 원하는 width
            target_height = 512  # 너가 원하는 height

            vis_img = cv2.resize(
                vis_img,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR
            )
            # ===========================================

            # FPS 표시
            cv2.putText(
                vis_img,
                f"FPS: {fps}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                vis_img,
                f"Resist: {sensitivity}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.imshow("Tactile Visualization", vis_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('e'):  # +5
                sensitivity = min(99, sensitivity + 5)
                env.set_resistance(sensitivity)

            elif key == ord('w'):  # -5
                sensitivity = max(0, sensitivity - 5)
                env.set_resistance(sensitivity)

            elif key == ord('q'):
                print("🛑 'q' pressed, exiting...")
                break
    except KeyboardInterrupt:
        print("🛑 Interrupted, closing...")

    finally:
        env.close()
        cv2.destroyAllWindows()
        print("Visualization closed.")


class DummyStage:
    """
    SensorEnv에서 stage를 요구하므로 dummy queue 로 대체
    """
    def empty(self):
        return True

    def get(self):
        return None


if __name__ == "__main__":
    # 사용 예시
    # ESP32 연결된 포트 리스트
    import multiprocessing as mp
    mp.set_start_method('fork', force=True)

    ports = [
        "/dev/tty.usbserial-01C640F9"
    ]

    visualize_tactile(
        ports=ports,
        stack_num=1,                 # 최근 1개 프레임만 보기
        adaptive_calibration=True,  # 보정 사용 X
        normalize=True
    )
