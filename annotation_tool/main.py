# coding=utf-8
import os
import re

from queue import Queue
from collections import deque

from depthai_utils import *


# Original videos:
# 0-center.mp4 00:28
# 1-center.mp4 02:35
# 2-center.mp4 00:18
# 4-center.mp4 00:33
# 5-center.mp4 00:17
# 7-center.mp4 01:27
# 8-center.mp4 00:18
# 9-center.mp4 00:16
# 11-center.mp4 00:11
# 12-center.mp4 00:31
# 13-center.mp4 01:43
# 14-center.mp4 00:56
# 15-center.mp4 01:04
# 16-center.mp4 00:42
# 17-center.mp4 00:36
# 18-center.mp4 01:32
# 19-center.mp4 00:35
# 21-center.mp4 01:19
# 22-center.mp4 02:48
#
# Videos:
# 0-center-1.mp4
# 1-center-1.mp4
# 1-center-2.mp4
# 1-center-3.mp4
# 1-center-4.mp4
# 1-center-5.mp4
# 2-center-1.mp4
# 4-center-1.mp4
# 5-center-1.mp4
# 7-center-1.mp4
# 7-center-2.mp4
# 7-center-3.mp4
# 8-center-1.mp4
# 9-center-1.mp4
# 11-center-1.mp4
# 12-center-1.mp4
# 13-center-1.mp4
# 13-center-2.mp4
# 13-center-3.mp4
# 13-center-4.mp4
# 14-center-1.mp4
# 14-center-2.mp4
# 15-center-1.mp4
# 15-center-2.mp4
# 16-center-1.mp4
# 16-center-2.mp4
# 17-center-1.mp4
# 18-center-1.mp4
# 18-center-2.mp4
# 18-center-3.mp4
# 19-center-1.mp4
# 21-center-1.mp4
# 21-center-2.mp4
# 21-center-3.mp4
# 22-center-1.mp4
# 22-center-2.mp4
# 22-center-3.mp4
# 22-center-4.mp4
# 22-center-5.mp4
# 22-center-6.mp4
#
# Cutting videos:
# ffmpeg -i 22-center.mp4 -ss 00:00:00 -t 00:00:29 -c copy 22-center-1.mp4
#
# Running script:
# python main.py -vid videos/22-center-1.mp4

class Main(DepthAI):
    def __init__(self, file=None, camera=False, record=False, annotate=False, play=False):
        self.filename = os.path.splitext(os.path.basename(file))[0]
        match = re.match(r"([0-9]+)\-([a-z]+)\-([0-9]+)", self.filename, re.I)
        try:
            if match:
                items = match.groups()
                self.prefix = "{:02d}".format(int(items[0])) + "{:02d}".format(int(items[2]))
            else:
                raise ValueError("Video format must be number-string-number")
        except:
            raise ValueError("Video format must be number-string-number")

        self.cam_size = (720, 1280)
        super(Main, self).__init__(file, camera)
        self.emo_frame = Queue()
        self.output = None
        if record:
            fourcc = cv2.VideoWriter_fourcc(*'xvid')
            self.output = cv2.VideoWriter(self.filename + '.avi', fourcc, 20.0, (1920, 1080))
        if annotate:
            self.annotate = True
            self.start = False
            self.classes = [ "non-stressed", "stressed" ]
            self.current_class = 0
            os.system(f"sed -i '/^{self.prefix}/d' data.txt")
            for label in self.classes:
                os.system(f"rm -r dataset/{label}/{self.prefix}* || true")
                (Path(__file__).parent / Path(f'dataset/{label}')).mkdir(parents=True, exist_ok=True)
            self.current_frame = 0
            self.text_file = open("data.txt", "a+")
        if play:
            self.annotate = False
            self.start = False
            self.classes = [ "non-stressed", "stressed" ]
            self.play = True
            with open("data.txt", "r") as f:
                self.saved_text_file = [ line.rstrip() for line in f if line.startswith(self.prefix) ]
            self.current_annotation_idx = 0
            self.current_annotation = [ int(i[-6:]) for i in self.saved_text_file[self.current_annotation_idx].split(",") ]
            self.current_frame = 0
            self.max_annotation = len(self.saved_text_file) - 1

        self.last_frames = deque()

    def check_annotation_and_next(self):
        if self.current_annotation is None:
            return None
        if self.current_annotation_idx == self.max_annotation:
            current_class = self.current_annotation[1]
            self.current_annotation = None
            return current_class
        elif self.current_frame == self.current_annotation[0]:
            current_class = self.current_annotation[1]
            self.current_annotation_idx += 1
            self.current_annotation = [ int(i[-6:]) for i in self.saved_text_file[self.current_annotation_idx].split(",") ]
            return current_class
        return None

    def create_nns(self):
        self.create_nn("models/face-detection-retail-0004.blob", "face")
        self.create_nn("models/emotions-recognition-retail-0003.blob", "emo")

    def start_nns(self):
        self.face_in = self.device.getInputQueue("face_in")
        self.face_nn = self.device.getOutputQueue("face_nn")
        self.emo_in = self.device.getInputQueue("emo_in")
        self.emo_nn = self.device.getOutputQueue("emo_nn")

    def run_face(self):
        data, scale, top, left = to_planar(self.frame, (300, 300))
        nn_data = run_nn(
            self.face_in,
            self.face_nn,
            {"data": data},
        )

        if nn_data is None:
            return False

        results = to_bbox_result(nn_data)

        self.face_coords = [
            frame_norm((300, 300), *obj[3:7]) for obj in results if obj[2] > 0.8
        ]

        self.face_num = len(self.face_coords)
        if self.face_num == 0:
            return False

        self.face_coords = restore_point(self.face_coords, scale, top, left).astype(int)

        self.face_coords = scale_bboxes(self.face_coords, scale=True, scale_size=1.5)

        for i in range(self.face_num):
            face = self.frame[
                self.face_coords[i][1] : self.face_coords[i][3],
                self.face_coords[i][0] : self.face_coords[i][2],
            ]
            self.emo_frame.put(face)
        if debug:
            bbox = self.face_coords[0]
            self.draw_bbox(bbox, (10, 245, 10))

        if hasattr(self, 'annotate'):
            if self.annotate and self.face_num > 0:
                timestamp = int(time.time() * 10000)
                det_frame = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                image_format = self.prefix + "{:06d}".format(self.current_frame)
                cropped_path = f'dataset/{self.classes[self.current_class]}/{image_format}.jpg'
                cv2.imwrite(cropped_path, det_frame)
                self.text_file.write(str(image_format) + "," + str(self.current_class) + "\r\n")

        return True

    def run_emo(self):
        for i in range(self.face_num):
            emo_frame = self.emo_frame.get()
            emo = ["neutral", "happy", "sad", "surprise", "anger"]
            nn_data = run_nn(
                self.emo_in,
                self.emo_nn,
                {"prob": to_planar(emo_frame, (64, 64))[0]},
            )
            if nn_data is None:
                return
            out = to_nn_result(nn_data)
            # print(out)
            emo_r = emo[out.argmax()]

            self.last_frames.append(emo_r)
            if len(self.last_frames) > 50:
                self.last_frames.popleft()
            emo_count = [ self.last_frames.count(emo[i]) for i in range(5) ]
            top_emo = emo[emo_count.index(max(emo_count))]

            self.put_text(
                top_emo,
                (self.face_coords[i][0], self.face_coords[i][1] + 80),
                (0, 0, 255),
            )

            if self.output is not None:
                self.output.write(self.debug_frame)

    def parse_fun(self):
        face_success = self.run_face()
        if face_success:
            if hasattr(self, 'annotate'):
                if self.annotate:
                    if self.start:
                        self.annotate_video()
                    if not self.start:
                        self.start = True
            elif not hasattr(self, 'play'):
                self.run_emo()
        if hasattr(self, 'play'):
            if self.play:
                current_annotation = self.check_annotation_and_next()
                if current_annotation is not None:
                    self.put_text(
                        "Saved annotation: " + self.classes[current_annotation],
                        (10, 80),
                        (0, 0, 255),
                    )
        if hasattr(self, 'play') or hasattr(self, 'annotate'):
            self.current_frame += 1

    def annotate_video(self):
        key = cv2.waitKey(1)

        if key == ord('n'):
            self.current_class = 0
        elif key == ord('s'):
            self.current_class = 1
        elif key == ord("q"):
            cv2.destroyAllWindows()
            self.fps.stop()
            print(f"FPS: {self.fps.fps():.2f}")
            raise StopIteration()

        self.put_text(
            "Annotating: " + self.classes[self.current_class],
            (10, 80),
            (0, 0, 255),
        )

    def __del__(self):
        if self.output is not None:
            self.output.release()
        if hasattr(self, 'annotate'):
            if self.annotate and self.text_file is not None:
                self.text_file.close()


if __name__ == "__main__":
    if args.video:
        Main(file=args.video, annotate=args.annotate, play=args.play).run()
    else:
        Main(camera=args.camera).run()

