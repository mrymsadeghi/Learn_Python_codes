#pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 pixellib opencv-python

import pixellib
# from pixellib.torchbackend.instance import instance_segmentation
from pixellib.instance import instance_segmentation
import cv2

segmentation_model = instance_segmentation()
segmentation_model.load_model('mask_rcnn_coco.h5')


# segmentation_model.segmentImage("input/dogs_n_cars.jpg", output_image_name="output/instance_dogs_n_cars.jpg",
#                            text_size=2, box_thickness=2, text_thickness=2,
#                            show_bboxes=True)



# Webcam
# capture = cv2.VideoCapture(0)
# segmentation_model.process_camera(capture, frames_per_second= 15, output_video_name="output/output.mp4", show_frames=True,
#                              show_bboxes=True,
#                              frame_name="frame",
#                              extract_segmented_objects=False,
#                              save_extracted_objects=False)




# Video
segmentation_model.process_video("input/test_vid2.mp4", frames_per_second= 15, output_video_name="output/seg_video.mp4")












""" 
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Apply instance segmentation
    segmask, output  = segmentation_model.segmentFrame(frame, show_bboxes=True)

    cv2.imshow('Instance Segmentation', output)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()




"""      """         """        
