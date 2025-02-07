import numpy as np
import cv2
import mediapipe as mp
import joblib
import time
import os, sys
from random import random
from english_words import english_words_lower_alpha_set
from flask import Flask, render_template, Response, request



global easy, medium, hard, freestyle, switch
easy=1
medium=0
hard=0
freestyle=0
switch=1

clf = joblib.load("random_forest.joblib")



app = Flask(__name__, template_folder='./template')


'''
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    results = model.process(image)                 # Make prediction
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results
    
def get_landmark_dist_test(results, x, y):
    hand_array = []
    wrist_pos = results.multi_hand_landmarks[0].landmark[0]
    for result in results.multi_hand_landmarks[0].landmark:
        hand_array.append((result.x-wrist_pos.x) * (width/x))
        hand_array.append((result.y-wrist_pos.y) * (height/y))
    return(hand_array[2:])

def camera_max():
    #Returns int value of available camera devices connected to the host device
    camera = 0
    while True:
        if (cv2.VideoCapture(camera).grab()):
            camera = camera + 1
        else:
            cv2.destroyAllWindows()
            return(max(0,int(camera-1)))

words = [i for i in sorted(list(english_words_lower_alpha_set)) if 'z' not in i and len(i) > 3 and len(i) <= 10]
start_time = time.time()
curr_time = 0
easy_word_user = ''
eraser = 0
easy_word = words[int(random()*len(words))].upper()
easy_word_index = 0
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

#Main function
cap = cv2.VideoCapture(camera_max())
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set mediapipe model
mp_hands = mp.solutions.hands # Hands model
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
    while cap.isOpened():
 
        # Read feed
        ret, frame = cap.read()

        try:
            cv2.putText(frame, easy_word, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_4)
            cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
        except Exception as e:
            print(e)
 
        # Make detections
        image, results = mediapipe_detection(frame, hands)
        
        letter_help = cv2.resize(cv2.imread('easy_mode_letters/{}.png'.format(easy_word[easy_word_index].lower())), (0,0), fx=0.2, fy=0.2)
         
        #Find bounding box of hand
        if results.multi_hand_landmarks:
            x = [None,None]
            y=[None,None]
            for result in results.multi_hand_landmarks[0].landmark:
                if x[0] is None or result.x < x[0]: x[0] = result.x
                if x[1] is None or result.x > x[1]: x[1] = result.x

                if y[0] is None or result.y < y[0]: y[0] = result.y
                if y[1] is None or result.y > y[1]: y[1] = result.y

            
            if curr_time < round((time.time() - start_time)/3,1) and x[0] is not None:
                    curr_time = round((time.time() - start_time)/3,1)
                    try:
                        test_image = get_landmark_dist_test(results, x[1]-x[0], y[1]-y[0])
                        test_pred = np.argmax(clf.predict_proba(np.array([test_image])))
                        test_probs = clf.predict_proba(np.array([test_image]))[0]
                        if max(test_probs) >= 0.8 or (max(test_probs) >= 0.6 and letters[test_pred] in ['r','v']):
                            pred_letter = letters[test_pred].upper()
                            if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and (easy_word_index == 0 or easy_word[easy_word_index] != easy_word[easy_word_index - 1]):
                                easy_word_user += pred_letter
                                easy_word_index += 1
                                location = results.multi_hand_landmarks[0].landmark[0].x
                            if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and easy_word_index > 0 and easy_word[easy_word_index] == easy_word[easy_word_index - 1] and abs(location - results.multi_hand_landmarks[0].landmark[0].x) > 0.1:
                                easy_word_user += pred_letter
                                easy_word_index += 1
                                location = results.multi_hand_landmarks[0].landmark[0].x
                        
                        if easy_word_user == easy_word:
                            time.sleep(0.5)
                            easy_word = words[int(random()*len(words))].upper()
                            easy_word_index = 0
                            easy_word_user = ''
                            
                    except Exception as e:
                        print(e)
                
        # Show to screen
        frame[5:5+letter_help.shape[0],width-5-letter_help.shape[1]:width-5] = letter_help
        cv2.imshow('OpenCV Feed', frame)
        

        # Break gracefully
        key = cv2.waitKey(20)
        if key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()'''



def camera_max():
    '''Returns int value of available camera devices connected to the host device'''
    camera = 0
    while True:
        if (cv2.VideoCapture(camera).grab()):
            camera = camera + 1
        else:
            cv2.destroyAllWindows()
            return(max(0,int(camera-1)))
        
cam_max = camera_max()

cap = cv2.VideoCapture(cam_max, cv2.CAP_DSHOW)


letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
words = [i for i in sorted(list(english_words_lower_alpha_set)) if 'z' not in i and len(i) > 3 and len(i) <= 10]
start_time = time.time()
curr_time = 0
easy_word_user = ''
eraser = 0
easy_word = words[int(random()*len(words))].upper()
easy_word_index = 0
location = 0
letter_help = 0


def easy_mode(frame):
    global cap, easy_word_user, easy_word, easy_word_index, curr_time, location, letter_help
    
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        results = model.process(image)                 # Make prediction
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
        return image, results

    def get_landmark_dist_test(results, x, y):
        hand_array = []
        wrist_pos = results.multi_hand_landmarks[0].landmark[0]
        for result in results.multi_hand_landmarks[0].landmark:
            hand_array.append((result.x-wrist_pos.x) * (width/x))
            hand_array.append((result.y-wrist_pos.y) * (height/y))
        return(hand_array[2:])


    #Main function
    #cap = cv2.VideoCapture(cam_max)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set mediapipe model
    mp_hands = mp.solutions.hands # Hands model
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while cap.isOpened():

            # Read feed
            #ret, frame = cap.read()

            try:
                cv2.putText(frame, easy_word, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
            except Exception as e:
                print(e)

            # Make detections
            image, results = mediapipe_detection(frame, hands)

            letter_help = cv2.resize(cv2.imread('easy_mode_letters/{}.png'.format(easy_word[easy_word_index].lower())), (0,0), fx=0.2, fy=0.2)

            #Find bounding box of hand
            if results.multi_hand_landmarks:
                x = [None,None]
                y=[None,None]
                for result in results.multi_hand_landmarks[0].landmark:
                    if x[0] is None or result.x < x[0]: x[0] = result.x
                    if x[1] is None or result.x > x[1]: x[1] = result.x

                    if y[0] is None or result.y < y[0]: y[0] = result.y
                    if y[1] is None or result.y > y[1]: y[1] = result.y


                if curr_time < round((time.time() - start_time)/3,1) and x[0] is not None:
                        curr_time = round((time.time() - start_time)/3,1)
                        try:
                            test_image = get_landmark_dist_test(results, x[1]-x[0], y[1]-y[0])
                            test_pred = np.argmax(clf.predict_proba(np.array([test_image])))
                            test_probs = clf.predict_proba(np.array([test_image]))[0]
                            print("Predicted:",letters[test_pred], ", pred prob:", max(test_probs), ", current index:", easy_word_index, ", current time:", curr_time)
                            if max(test_probs) >= 0.8 or (max(test_probs) >= 0.6 and letters[test_pred] in ['p','r','u','v']):
                                pred_letter = letters[test_pred].upper()
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and (easy_word_index == 0 or easy_word[easy_word_index] != easy_word[easy_word_index - 1]):
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and easy_word_index > 0 and easy_word[easy_word_index] == easy_word[easy_word_index - 1] and abs(location - results.multi_hand_landmarks[0].landmark[0].x) > 0.1:
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x

                            if easy_word_user == easy_word:
                                time.sleep(0.5)
                                easy_word = words[int(random()*len(words))].upper()
                                easy_word_index = 0
                                easy_word_user = ''

                        except Exception as e:
                            print(e)

            # Show letter helper
            frame[5:5+letter_help.shape[0],width-5-letter_help.shape[1]:width-5] = letter_help

            return frame
            
    return frame


def medium_mode(frame):
    global cap, easy_word_user, easy_word, easy_word_index, curr_time, location, letter_help
    
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        results = model.process(image)                 # Make prediction
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
        return image, results

    def get_landmark_dist_test(results, x, y):
        hand_array = []
        wrist_pos = results.multi_hand_landmarks[0].landmark[0]
        for result in results.multi_hand_landmarks[0].landmark:
            hand_array.append((result.x-wrist_pos.x) * (width/x))
            hand_array.append((result.y-wrist_pos.y) * (height/y))
        return(hand_array[2:])
    
    #Main function
    #cap = cv2.VideoCapture(cam_max)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set mediapipe model
    mp_hands = mp.solutions.hands # Hands model
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while cap.isOpened():

            # Read feed
            #ret, frame = cap.read()

            try:
                cv2.putText(frame, easy_word, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
            except Exception as e:
                print(e)

            # Make detections
            image, results = mediapipe_detection(frame, hands)

            #Find bounding box of hand
            if results.multi_hand_landmarks:
                x = [None,None]
                y=[None,None]
                for result in results.multi_hand_landmarks[0].landmark:
                    if x[0] is None or result.x < x[0]: x[0] = result.x
                    if x[1] is None or result.x > x[1]: x[1] = result.x

                    if y[0] is None or result.y < y[0]: y[0] = result.y
                    if y[1] is None or result.y > y[1]: y[1] = result.y


                if curr_time < round((time.time() - start_time)/3,1) and x[0] is not None:
                        curr_time = round((time.time() - start_time)/3,1)
                        try:
                            test_image = get_landmark_dist_test(results, x[1]-x[0], y[1]-y[0])
                            test_pred = np.argmax(clf.predict_proba(np.array([test_image])))
                            test_probs = clf.predict_proba(np.array([test_image]))[0]
                            print("Predicted:",letters[test_pred], ", pred prob:", max(test_probs), ", current index:", easy_word_index, ", current time:", curr_time)
                            if max(test_probs) >= 0.8 or (max(test_probs) >= 0.6 and letters[test_pred] in ['p','r','u','v']):
                                pred_letter = letters[test_pred].upper()
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and (easy_word_index == 0 or easy_word[easy_word_index] != easy_word[easy_word_index - 1]):
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and easy_word_index > 0 and easy_word[easy_word_index] == easy_word[easy_word_index - 1] and abs(location - results.multi_hand_landmarks[0].landmark[0].x) > 0.1:
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x

                            if easy_word_user == easy_word:
                                time.sleep(0.5)
                                easy_word = words[int(random()*len(words))].upper()
                                easy_word_index = 0
                                easy_word_user = ''

                        except Exception as e:
                            print(e)

            try: 
                letter_help == 0
            except:
                frame[5:5+letter_help.shape[0],width-5-letter_help.shape[1]:width-5] = frame[5:5+letter_help.shape[0],width-5-letter_help.shape[1]:width-5]
            
            return frame
            
    return frame


def sign_frame():  # generate frame by frame from camera
    global easy, cap
    while True:
        success, frame = cap.read() 
        if success:
            if(easy):                
                frame = easy_mode(frame)
            elif(medium):                
                frame = medium_mode(frame)
   
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    return Response(sign_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/requests',methods=['POST','GET'])
def mode():
    global switch, easy, medium, hard, free
    if request.method == 'POST':
        if request.form.get('easy') == 'Easy':
            easy= not easy
            medium, hard, free =  0, 0, 0
        elif  request.form.get('medium') == 'Medium':
            medium=not medium
            medium, hard, free =  0, 0, 0
        elif  request.form.get('hard') == 'Hard':
            hard=not hard
            medium, hard, free =  0, 0, 0
        elif  request.form.get('free') == 'Freestyle':
            free=0  
            medium = 0
            hard = 0
        '''elif  request.form.get('switch') == 'Stop/Start':
            
            if switch:
                switch=0
                cap.release()
                cv2.destroyAllWindows()
                
            else:
                cap = cv2.VideoCapture(camera_max())
                switch=1'''
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')



# Your routes and other functions...

from waitress import serve
from sign_app import app  # Import your Flask app from the sign_app file

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)  # Run the app on port 5000




