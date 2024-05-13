# detect.py
import cv2
import time
from playsound import playsound
# from test_sound import sound_1
import pygame
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import smtplib


from flask import Flask, render_template

l = ['_', 'bala', 'guna','selvam']
detected_persons = []




def report_send_mail(image_path):
    with open(image_path, 'rb') as f:
        img_data = f.read()
        label="Aleart"
    fromaddr = "sonabala827@gmail.com"
    toaddr = 'guna3siva@gmail.com'
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Alert"
    body = label
    msg.attach(MIMEText(body, 'plain'))  # attach plain text
    image = MIMEImage(img_data, name=os.path.basename(image_path))
    msg.attach(image)  # attach image
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(fromaddr, "hlkoquxeydxkvwmn")
    text = msg.as_string()
    s.sendmail(fromaddr, toaddr, text)
    s.quit()
def audio():
        pygame.mixer.init()
        pygame.mixer.music.load("buzzer.mpeg")
        pygame.mixer.music.play()

#detected_persons = []
def detect():
    capture_duration = 10
    faceDetect = cv2.CascadeClassifier('MTCN.xml')
    cam = cv2.VideoCapture(0)
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer/trainingData.yml")
    ret, img = cam.read()
    start_time = time.time()
    detected_persons = []
    processed_ids = set()  # Set to keep track of processed IDs

    while int(time.time() - start_time) < capture_duration:
        _, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        font=cv2.FONT_HERSHEY_PLAIN
        #detected_persons = []

        for (x, y, w, h) in faces:
                cv2.rectangle(img, (x - 50, y - 50), (x + w + 50, y + h + 50), (255, 0, 0), 2)
                id, conf = rec.predict(gray[y:y + h, x:x + w])
               
                print(conf)
                if conf < 60 : # and id not in processed_ids:
                    person_info = ''
                    if id == 1:
                        
                        print('bala')
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                        cv2.putText(img,'bala',(x,y+200),font,2,(0,0,255),2)
                        print('Door Accessed')
                       
                        time.sleep(2)
                        
                      
                    
                     
                    if id == 2:
                        
                        print('guna')
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                        cv2.putText(img,'guna',(x,y+200),font,2,(0,0,255),2)
                        print('Door Accessed')
                        time.sleep(2)

                    if id == 3:
                        
                        print('selvam')
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                        cv2.putText(img,'selvam',(x,y+200),font,2,(0,0,255),2)
                        print('Door Accessed')
                        time.sleep(2)

                    
                        
                        
          
                      
                    

                else:

                   print('unknown')
                   cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                   cv2.putText(img,'unknown',(x,y+200),font,2,(0,0,255),2)
                   print('Access Denied')    
                        
                   audio()

                   cv2.imwrite('image.jpg', img)
                   image_path = 'image.jpg'
                   label_ = id
                 
                   time.sleep(2)
                   report_send_mail(image_path)

                    
                                     
                    

                    

        cv2.imshow("Face", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return detected_persons

if __name__ == '__main__':
    detect()
    pass
