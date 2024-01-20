import cv2 as cv
from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
import torch

NEW_WIDTH = 900
NEW_HEIGHT = 700

def main(video_path, model, tracker, output_video_path):
    cap = cv.VideoCapture(video_path)

    fps = int(cap.get(cv.CAP_PROP_FPS))

    out = cv.VideoWriter(output_video_path,  
                         cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                         fps, (NEW_WIDTH, NEW_HEIGHT)) 

    while True:
        ok, image = cap.read()

        if not ok:
            print("Cannot read the video feed.")
            break

        image = cv.resize(image, (NEW_WIDTH, NEW_HEIGHT))

        bboxes, confidences, class_ids = model.detect(image)
        tracks = tracker.update(bboxes, confidences, class_ids)
        updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)

        updated_image = draw_tracks(updated_image, tracks)

        out.write(updated_image)

        cv.imshow("image", updated_image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Object detections in input video using YOLOv3 trained on COCO dataset.'
    )

    parser.add_argument(
        '--video', '-v', type=str, default="Sample Videos/Cars waiting for the elderly to cross.mp4", help='Input video path.')

    parser.add_argument(
        '--tracker', type=str, default= 'SORT',
        help="Tracker used to track objects. Options include ['CentroidTracker', 'CentroidKF_Tracker', 'SORT']")
    
    parser.add_argument(
        '--output_path', type=str, default= 'Results/output.mp4',
        help="Tracker used to track objects. Options include ['CentroidTracker', 'CentroidKF_Tracker', 'SORT']")

    args = parser.parse_args()

    if args.tracker == 'CentroidTracker':
        tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    elif args.tracker == 'CentroidKF_Tracker':
        tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
    elif args.tracker == 'SORT':
        tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
    else:
        raise NotImplementedError("Wrong params")
    

    model = YOLOv3(
        weights_path= "models/MOT-YOLOv3/yolov3.weights",
        configfile_path= "models/MOT-YOLOv3/yolov3.cfg",
        labels_path= "models/MOT-YOLOv3/coco_names.json",
        confidence_threshold= 0.8,
        nms_threshold= 0.2,
        draw_bboxes= True,
        use_gpu= True
    )
    print('cuda' if torch.cuda.is_available() else 'cpu')

    main(args.video, model, tracker, args.output_path)
