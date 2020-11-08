import tkinter
from tkinter.filedialog import askopenfilename
from tkinter import *
from PIL import Image, ImageTk, ImageDraw

import constants
from models import classification, object_detection
import numpy as np
import json
import random
from utils.files import JsonFile


class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

        # Load model
        self.detecting = object_detection.get_detecting_model()
        self.classifying = classification.get_classifying_model()

        self.bboxes = None
        self.label_to_name = JsonFile(constants.MAPPING).load()

    def draw_bb(self, image, bboxes):
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])),
                           outline='red', width=2)
        self.show_img(self.store_img_lable, image, 0, 0)

    def detect_bbox(self):
        self.bboxes = self.detecting.predict_bb(name=self.name)

    def btn_detect(self):

        # Important to copy image before detect
        image = self.image
        self.detect_bbox(image)
        bboxes = self.bboxes
        self.draw_bb(image, bboxes)

    def init_window(self):
        # changing the title of our master widget
        self.master.title("Demo")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        self.var = StringVar()
        self.label = Label(master=self.master, textvariable=self.var,
                           font=("Helvetica", 16))
        self.label.pack()

        self.open_file_btn = Button(master=self.master, command=self.btn_open,
                                    text='Open File')
        self.open_file_btn.pack()

        self.detect_btn = Button(master=self.master,
                                 command=self.btn_detect,
                                 text='Detect Image')
        self.detect_btn.pack()

        self.report_btn = Button(master=self.master,
                                 command=lambda: self.new_window(
                                        NewWin).pack(), text='Report')
        self.report_btn.pack()

        self.master.bind('<Motion>', self.motion)

        self.store_img_lable = Label(master=self.master)

        self.var = StringVar()
        self.reference_image_label = Label(master=self.master, compound=BOTTOM,
                                           textvariable=self.var,
                                           font=("Helvetica", 16))

    def new_window(self, _class):
        self.new = Toplevel(self.master)
        _class(self.new)

        self.report_label = Label(master=self.new)

        # filter bboxes in classifying
        self.report_bboxes = {}
        self.new_bb = []
        for box in self.bboxes:
            crop_image = self.image.crop((box[0], box[1], box[2], box[3]))
            crop_image = crop_image.resize(constants.IMAGE_SIZE)
            dist, ind = self.classifying.predict_class(crop_image)
            if self.label_to_name[str(ind[0][
                                          0])] not in self.report_bboxes.keys() and dist < constants.DISTANCE:
                self.report_bboxes[self.label_to_name[str(ind[0][0])]] = []
            if dist < constants.DISTANCE:
                self.report_bboxes[self.label_to_name[str(ind[0][0])]].append(
                    box)
                self.new_bb.append(box)

        # sort bboxes
        self.filter_bboxes = []
        for i in range(10):

            # find minbox
            minbox = [1024, 1024, 1024, 1024]
            for box in self.new_bb:
                if box in np.array(self.filter_bboxes):
                    continue
                if minbox[0] + minbox[1] > box[0] + box[1]:
                    minbox = box
            self.filter_bboxes.append(minbox)

            # find near box
            for box2 in self.new_bb:
                # random.shuffle(self.new_bb)
                lastbox = self.filter_bboxes[-1]
                distance2box = 1024
                nearbox = None
                s = 0
                if box2 in np.array(self.filter_bboxes):
                    continue
                for box1 in self.new_bb:
                    if box1 in np.array(self.filter_bboxes):
                        continue
                    if ((lastbox[1] < box1[1] and lastbox[3] > box1[1]) or (
                            lastbox[1] > box1[1] and lastbox[1] < box1[3])) and \
                            box1[0] - lastbox[2] < distance2box:
                        nearbox = box1
                        distance2box = box1[0] - lastbox[2]
                        s = 1
                if s == 1:
                    self.filter_bboxes.append(nearbox)

        # draw bboxes
        draw = ImageDraw.Draw(self.original_image)
        s = 1
        for key in self.report_bboxes:
            R = random.randint(0, 255)
            B = random.randint(0, 255)
            G = random.randint(0, 255)
            for box in self.report_bboxes[key]:
                draw.rectangle(((box[0], box[1]), (box[2], box[3])),
                               outline=(R, B, G), width=2)
        for box in self.filter_bboxes:
            draw.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                      text=str(s))
            s = s + 1
        self.show_img(self.report_label, self.original_image, 0, 0)

    def show_img(self, config, image, x, y):
        render = ImageTk.PhotoImage(image)
        # labels can be text or images
        config.config(image=render)
        config.image = render
        config.place(x=x, y=y)

    def open_file(self):
        name = askopenfilename(
            initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
            filetypes=(("Text File", "*.jpeg"), ("All Files", "*.*")),
            title="Choose a file."
        )
        self.name = name
        return name

    def read_image(self):
        load = Image.open(self.open_file())
        self.image = load
        self.original_image = self.image.copy()
        return load

    def btn_open(self):
        render = self.read_image()
        self.show_img(self.store_img_lable, render, 0, 0)

    def motion(self, event):
        if event.x > 1024 or event.y > 1024:
            return
        for box in self.bboxes:
            if event.x > box[0] and event.y > box[1] and event.x < box[
                2] and event.y < box[3]:
                crop_image = self.image.crop((box[0], box[1], box[2], box[3]))
                crop_image = crop_image.resize(constants.IMAGE_SIZE)
                dist, ind = self.classifying.predict_class(crop_image)

                name = self.label_to_name[str(ind[0][0])]
                image = Image.open(constants.REFERENCE_IMAGE_PATH + name)
                self.var.set(name + '\n' + str(dist[0][0]))
                self.show_img(self.reference_image_label, image, 1024, 0)
        else:
            return
        return


class NewWin:
    def __init__(self, root):
        self.root = root
        # self.windowframe = Window(root)


def main():
    root = tkinter.Tk()
    window = Window(root)
    root.mainloop()


if __name__ == '__main__':
    main()
