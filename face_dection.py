import cv2
import mediapipe as mp
# 찾은 얼굴에 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection # 얼굴 검출을 위한 face_detection 모듈을 사용
mp_drawing = mp.solutions.drawing_utils # 얼굴의 특징을 그리기 위한 drawing_utils0모듈을 사용


cap = cv2.VideoCapture('love.mp4') #model_selection => 0(1m이내의 얼굴) or 1 (5m이내의 얼굴) /min detection_cofidence => 얼굴 인식 퍼센트
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
#with 문 끝나면 자동으로 탈출
        # To improve performance, optionally mark the image as not writeable to
        # pass by qreference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# bgr이미지를 RGB로 변환
        results = face_detection.process(image) # 얼굴 검출 결과 반환

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.detections: #검출 정보 얼굴에 그리기 네모 점 같은 객체그리기
            # 6개 특징 : 오른쪽 눈, 왼쪽 눈, 코 끝부분, 입 중심, 오른쪽 귀, 왼쪽 귀 (귀구슬점, 이주)
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
            keypoints=detection.location_data.relative_keypoints
            right_eye=keypoints[0]
            left_eye=keypoints[1]
            nose_tip=keypoints[2]
            
            h,w,_=image.shape # 높이 넓이 가져오기
            right_eye=(int(right_eye.x*w),int(right_eye.y*h))
            left_eye=(int(left_eye.x*w),int(left_eye.y*h)) 
            nose_tip=(int(nose_tip.x*w),int(nose_tip.y*h))
            
            cv2.circle(image,right_eye,50,(255,0,0),10,cv2.LINE_AA)
            cv2.circle(image,left_eye,50,(0,255,0),10,cv2.LINE_AA)
            cv2.circle(image,nose_tip,50,(0,0,255),10,cv2.LINE_AA)
                        
        # Flip the image horizontally for a selfie-view display. 동영상이니까 flip 필요없음
        cv2.imshow('MediaPipe Face Detection',cv2.resize(image,None,fx=0.5,fy=0.5))
        
        if cv2.waitKey(1) == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
