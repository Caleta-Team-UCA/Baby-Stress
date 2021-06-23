import depthai as dai
import cv2
import typer
import numpy as np
from time import monotonic


def define_pipeline(face_detection_blob_path: str, stress_classifier_blob_path: str):
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)

    # Define sources
    face_detection = pipeline.createNeuralNetwork()
    face_detection.setBlobPath(face_detection_blob_path)

    image_manager = pipeline.createImageManip()
    image_manager.initialConfig.setResize(224, 224)

    stress_classifier = pipeline.createNeuralNetwork()
    stress_classifier.setBlobPath(stress_classifier_blob_path)

    # Define links
    input = pipeline.createXLinkIn()
    input.setStreamName("input")
    output = pipeline.createXLinkOut()
    output.setStreamName("output")

    # Linking
    input.out.link(face_detection.input)
    face_detection.out.link(image_manager.inputImage)
    image_manager.out.link(stress_classifier.input)
    stress_classifier.out.link(output.input)

    return pipeline


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def process_video(
    face_detection_blob_path: str,
    stress_classifier_blob_path: str,
    video_path: str,
):
    pipeline = define_pipeline(face_detection_blob_path, stress_classifier_blob_path)

    with dai.Device(pipeline) as device:
        q_in = device.getInputQueue(name="input")
        q_out = device.getOuputQueue(name="output")

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            read_correctly, frame = cap.read()

            if not read_correctly:
                break

            img = dai.ImgFrame()
            img.setData(to_planar(frame, (300, 300)))
            img.setTimestamp(monotonic())
            img.setWidth(300)
            img.setHeight(300)
            q_in.send(frame)

            output = q_out.get()

            print(output)

            if cv2.waitKey(1) == ord("q"):
                break


def main(
    face_detection_blob_path: str = "/home/users/ucadatalab_group/javierj/Baby-Stress/.Models/face_detection.blob",
    stress_classifier_blob_path: str = "/home/users/ucadatalab_group/javierj/Baby-Stress/.Models/resnet_imagenet/frozen_graph.blob",
    video_path: str = "/home/users/ucadatalab_group/javierj/SHARED/baby-stress/videos/21-center-3.mp4",
):
    process_video(face_detection_blob_path, stress_classifier_blob_path, video_path)


if __name__ == "__main__":
    typer.run(main)
