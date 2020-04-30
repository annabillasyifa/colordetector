import os
import kivy
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
# to use buttons:
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.logger import Logger
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import ScreenManager

kivy.require("1.10.1")


def colordetector():
    # capturing video through webcam
    cap = cv2.VideoCapture(0)

    while(1):
        _, img = cap.read()

        # converting frame(img i.e BGR) to HSV (hue-saturation-value)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        white_lower = np.array([0, 0, 0], np.uint8)
        white_upper = np.array([50, 50, 50], np.uint8)

        black_lower = np.array([205, 205, 205], np.uint8)
        black_upper = np.array([255, 255, 255], np.uint8)

        # defining the range of red color
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)

        # defining the Range of Blue color
        blue_lower = np.array([99, 115, 150], np.uint8)
        blue_upper = np.array([110, 255, 255], np.uint8)

        # defining the Range of yellow color
        yellow_lower = np.array([22, 60, 200], np.uint8)
        yellow_upper = np.array([60, 255, 255], np.uint8)

        # finding the range of red,blue and yellow color in the image
        red = cv2.inRange(hsv, red_lower, red_upper)
        blue = cv2.inRange(hsv, blue_lower, blue_upper)
        yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        black = cv2.inRange(hsv, white_lower, white_upper)
        white = cv2.inRange(hsv, black_lower, black_upper)

        # Morphological transformation, Dilation
        kernal = np.ones((5, 5), "uint8")

        red = cv2.dilate(red, kernal)
        res = cv2.bitwise_and(img, img, mask=red)

        blue = cv2.dilate(blue, kernal)
        res1 = cv2.bitwise_and(img, img, mask=blue)

        yellow = cv2.dilate(yellow, kernal)
        res2 = cv2.bitwise_and(img, img, mask=yellow)

        black = cv2.dilate(black, kernal)
        res3 = cv2.bitwise_and(img, img, mask=black)

        white = cv2.dilate(white, kernal)
        res4 = cv2.bitwise_and(img, img, mask=white)

        # Tracking the black Color
        (contours, hierarchy) = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv2.putText(img, "Black color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))

        # Tracking the white Color
        (contours, hierarchy) = cv2.findContours(white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(img, "White color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))

        # Tracking the Red Color
        (contours, hierarchy) = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, "Red color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

        # Tracking the Blue Color
        (contours, hierarchy) = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, "Blue color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))

        # Tracking the yellow Color
        (contours, hierarchy) = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Yellow  color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

                # cv2.imshow("Redcolour",red)
        cv2.imshow("Color Tracking", img)
        # cv2.imshow("red",res)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break





class ConnectPage(GridLayout):
    # runs on initialization
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inside = GridLayout()
        self.inside.cols = 2
        self.cols = 2
        self.join = Button(text="START")
        self.join.bind(on_press=self.color_detector)
        self.add_widget(Label())  # just take up the spot.
        self.add_widget(self.join)




    def color_detector(self, instance):
        return colordetector()


class PendeteksiWarna(App):
    def build(self):
        return ConnectPage()


if __name__ == "__main__":
    PendeteksiWarna().run()