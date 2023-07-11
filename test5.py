import cv2 
import numpy as np 
from imutils.object_detection import non_max_suppression 


# 'in.avi'是測試用的影片，將這邊改成 0 就可以用鏡頭了 
filename = 'test_video/in.avi' 
file_size = (1920,1080) 
scale_ratio = 1


def main():
  hog = cv2.HOGDescriptor()
  hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  
  
  cap = cv2.VideoCapture(filename)

  while cap.isOpened():
    success, frame = cap.read()
    if success:
      width = int(frame.shape[1] * scale_ratio)
      height = int(frame.shape[0] * scale_ratio)
      frame = cv2.resize(frame, (width, height))
             
      orig_frame = frame.copy()
      (bounding_boxes, weights) = hog.detectMultiScale(frame, 
                                                       winStride=(16, 16),
                                                       padding=(4, 4), 
                                                       scale=1.05)
      for (x, y, w, h) in bounding_boxes: 
        cv2.rectangle(orig_frame, 
          (x, y),  
          (x + w, y + h),  
          (0, 0, 255), 
          2)

      bounding_boxes = np.array([[x, y, x + w, y + h] for (
                                x, y, w, h) in bounding_boxes])
             
      selection = non_max_suppression(bounding_boxes, 
                                      probs=None, 
                                      overlapThresh=0.45)
         
      for (x1, y1, x2, y2) in selection:
        cv2.rectangle(frame, 
                     (x1, y1), 
                     (x2, y2), 
                     (0, 255, 0), 
                      4)
        
      cv2.imshow("Frame", frame)

      # 按下 q 可以離開畫面    
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    else:
      break

  # 釋放
  cap.release()
  cv2.destroyAllWindows() 


main()