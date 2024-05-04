
from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import random
from sklearn.cluster import KMeans
from operator import itemgetter
import pandas as pd


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/frame0")



def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def Read_from_DAP_AN(DAP_AN_path = './DAP_AN.csv'):
    dict = pd.read_csv(DAP_AN_path, header=None, index_col=0).to_dict()
    answer_key_ABCD = dict[1]
    print(answer_key_ABCD)
    answer_key = {}
    convert = {"A": 0, "B": 1, "C": 2, "D": 3}
    for key, val in answer_key_ABCD.items():
        answer_key[key] = convert[val]
    print(answer_key)
    return answer_key
global ANSWER_KEY
ANSWER_KEY = Read_from_DAP_AN()

def kMeans_predict_line( character_coor_list, n_clusters=5):
    SumY = np.array([0, 0, 0, 0, 0])
    NumY = np.array([0, 0, 0, 0, 0])
    Lines = [[], [], [], [], []]

    x1y1PointList= [[1, x1y1[1]] for x1y1 in character_coor_list]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x1y1PointList)
    pred_label = kmeans.predict(x1y1PointList)
    labels = np.unique(pred_label)
    for char_coor, pred in zip (character_coor_list, pred_label):
        for i in range(0, n_clusters):
            if pred == i:
                SumY[i] += char_coor[3]
                NumY[i] += 1
                Lines[i].append(char_coor)

    averageY = SumY / NumY

    for i in range(0, len(Lines)):
        Lines[i].append(averageY[i])

    SortedLines = sorted(Lines, key=lambda x: x[-1], reverse=False)
    SortedLines_isRemoved_Y = [s[:4] for s in SortedLines]

    return SortedLines_isRemoved_Y

def arrange_Contours( character_coor_list):
        character_coor_list = kMeans_predict_line(character_coor_list, n_clusters=5)
        print(character_coor_list)
        character_coor_list_arrangebyX = []
        for character_coor_arrangebyX in character_coor_list:
            character_coor_arrangebyX = sorted(character_coor_arrangebyX, key=itemgetter(0), reverse=False)
            character_coor_list_arrangebyX.append(character_coor_arrangebyX)
        # character_coor_list.sort(key=lambda x:x[0], reverse=False)
        return character_coor_list_arrangebyX

def FindAnswerIsCheck(thresh, boxes):
    isCheckAnswer = -1
    sum_total = 0
    number_of_check_answer = 0
    print(boxes)
    for i, box in enumerate(boxes):
        print(box)
        answer_thresh = thresh[box[1]:box[3], box[0]:box[2]]
        sum_total += cv2.countNonZero(answer_thresh)
    avg_total = sum_total / len(boxes)
    for i, box in enumerate(boxes):
        answer_thresh = thresh[box[1]:box[3], box[0]:box[2]]
        total = cv2.countNonZero(answer_thresh)
        print("total: ", total)
        if total >= avg_total:
            minus = total - avg_total
            if minus > 50:
                number_of_check_answer += 1
                isCheckAnswer = i
    if number_of_check_answer == 1:
        return isCheckAnswer
    else:
        return -1

def ScoreAnswer(answer_key, answer_from_student):
    score = 0
    number_of_correct_answer = 0
    for key, value in answer_key.items():
        if answer_from_student[key] == value:
            number_of_correct_answer += 1
    score = number_of_correct_answer / len(answer_key) * 10
    return score

def Chamdiem(img):
    '''
    input: image cv2
    return: draw is display image, score
    '''
    global ANSWER_KEY


    # answer_key = {0: 1, 1: 2, 2: 0, 3: 3, 4: 1}
    answer_from_student ={}
    draw = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    answer_coor_list = [] #Biến này dùng để chứa bounding box của vị trí nơi tô đáp án

        # Loop over the contours
    for c in cnts:
        # Compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # In order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and have an aspect ratio approximately equal to 1
        if w >= 20 and h >= 20 and ar >= 0.7 and ar <= 1.2 and x >= 50:
            # cv2.drawContours(image=draw, contours=[c], contourIdx=-1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            answer_coor_list.append([x, y, x+w, y+w])

        # Sort the question contours top-to-bottom
    answer_coor_list_arange_byXY = arrange_Contours(answer_coor_list)
    for i, boxes in enumerate(answer_coor_list_arange_byXY):
        # print(boxes)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        CheckAnswerIndex = FindAnswerIsCheck(thresh, boxes)

        answer_from_student[i+1] = CheckAnswerIndex
        print("ANSWER_KEY[1]: ", ANSWER_KEY[1])
        cv2.rectangle(draw, (boxes[ANSWER_KEY[i+1]][0], boxes[ANSWER_KEY[i+1]][1]), (boxes[ANSWER_KEY[i+1]][2], boxes[ANSWER_KEY[i+1]][3]), color=(255, 255, 0), thickness=2) 

    print(answer_from_student)
    #Tính điểm
    # cv2.putText(draw, "{:.1f}".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return draw, answer_from_student

# Hàm để mở hình ảnh và hiển thị trên giao diện
global processed_image
global CURRENT_PATH
global CURRENT_IMG
def open_image():
    global CURRENT_PATH
    global CURRENT_IMG

    # Open a file dialog to select an image
    path = filedialog.askopenfilename(initialdir="../Data", title="Select A File", filetype=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    # Ensure a file path was selected
    CURRENT_PATH = path
    if CURRENT_PATH:
        print(path)
        # Load the image and convert to a format that tkinter can handle
        img = cv2.imread(path)
        CURRENT_IMG = img
        image = Image.open(path)
        resized_image = image.resize((450, 500))
        # image.thumbnail((450, 500))  # Resize to fit the display area
        photo = ImageTk.PhotoImage(resized_image)

        # Set the image to the label
        canvas.itemconfig(answer_sheet_label, image=photo)
        canvas.resized_image=photo

def hienthi():
    # global score
    global ANSWER_KEY
    global CURRENT_IMG
    ANSWER_KEY = Read_from_DAP_AN()
    processed_image, answer_from_student = Chamdiem(CURRENT_IMG)

    score = ScoreAnswer(answer_key=ANSWER_KEY, answer_from_student=answer_from_student)

    # Convert processed image for tkinter
    processed_image_PIL = Image.fromarray(processed_image)
    resized_processed_image_PIL = processed_image_PIL.resize((450, 500))

    processed_photo = ImageTk.PhotoImage(image=resized_processed_image_PIL)

    # Display processed image and score
    canvas.itemconfig(result_label, image=processed_photo)
    canvas.resized_processed_image_PIL = processed_photo
    canvas.itemconfig(score_label, text=f"{score}")

    messagebox.showinfo("Info", "Đã chấm điểm!")



window = Tk()

window.geometry("1000x745")
window.configure(bg = "#F6BFC6")


canvas = Canvas(
    window,
    bg = "#F6BFC6",
    height = 745,
    width = 1000,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
ThemDapAn_img = PhotoImage(
    file=relative_to_assets("button_1.png"))
ThemDapAn_Btn = Button(
    image=ThemDapAn_img,
    borderwidth=0,
    highlightthickness=0,
    command=open_image,
    relief="flat"
)
ThemDapAn_Btn.place(
    x=273.0,
    y=671.0,
    width=100.0,
    height=65.0
)

ChamDiem_img = PhotoImage(
    file=relative_to_assets("button_2.png"))
ChamDiem_Btn = Button(
    image=ChamDiem_img,
    borderwidth=0,
    highlightthickness=0,
    command=hienthi,
    relief="flat"
)
ChamDiem_Btn.place(
    x=386.0,
    y=671.0,
    width=100.0,
    height=65.0
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    500.0,
    40.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    46.0,
    40.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    733.0,
    121.0,
    image=image_image_3
)

answer_sheet_img = PhotoImage(
    file=relative_to_assets("image_4.png"))
answer_sheet_label = canvas.create_image(
    261.0,
    408.0,
    image=answer_sheet_img
)

result_img = PhotoImage(
    file=relative_to_assets("image_5.png"))
result_label = canvas.create_image(
    733.0,
    408.0,
    image=result_img
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    261.0,
    121.0,
    image=image_image_6
)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    793.0,
    703.0,
    image=image_image_7
)

canvas.create_text(
    131.99999429658055,
    104.0,
    anchor="nw",
    text="Phiếu điểm học sinh",
    fill="#000000",
    font=("Inter Bold", 24 * -1)
)

canvas.create_text(
    617.0,
    104.0,
    anchor="nw",
    text="Kết quả chấm điểm",
    fill="#000000",
    font=("Inter Bold", 24 * -1)
)

canvas.create_text(
    508.0000095553696,
    687.0,
    anchor="nw",
    text="Điểm số: ",
    fill="#000000",
    font=("Inter Bold", 24 * -1)
)

score_label = canvas.create_text(
    634.9999790377915,
    686.0,
    anchor="nw",
    text="XX",
    fill="#000000",
    font=("Inter Bold", 24 * -1)
)
window.resizable(False, False)
window.mainloop()
