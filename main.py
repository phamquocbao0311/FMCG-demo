import tkinter
from tkinter.filedialog import askopenfilename
from tkinter import *
from PIL import Image, ImageTk, ImageDraw
import model
import classifying_model
import numpy as np
import json
import random

size_image = (80,80)
referencce_image_path = 'D:\\product_image\\images\\perfect_store_images\\'
mapping = './data/mapping.json'
distance = 0.7 #distance in classifying
pixel = 50
OModel_path = './data/object_detection/resnet50_csv_06.h5'
CModel_path = './data/classification/Arcface.h5'
dir_cluster = './data/cluster.npy'

class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

        self.store_img = Label(master=master)

        self.var = StringVar()
        self.reference_image = Label(master= master, compound = BOTTOM, textvariable = self.var, font=("Helvetica", 16))
        self.bboxes = None

        with open(mapping) as f:
            self.label_to_name = json.load(f)

        self.model = model.Model(OModel_path)
        self.classifying = classifying_model.Classifying(CModel_path, dir_cluster)

    def draw_bb(self, image, bboxes):
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='red', width=2)
        self.showImg(self.store_img, image, 0, 0)

    def detect_bbox(self, name):
        self.bboxes = self.model.predict_bb(name = self.name)

    def btn_detect(self):
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
        self.label = Label(master=self.master, textvariable = self.var, font=("Helvetica", 16))
        self.label.pack()

        self.button = Button(master=self.master, command  = self.btn_open, text = 'Open File')
        self.button.pack()

        self.button_detect = Button(master = self.master, command = self.btn_detect, text = 'Detect Image')
        self.button_detect.pack()

        self.button_report = Button(master=self.master, command=lambda: self.new_window(NewWin).pack(), text='Report')
        self.button_report.pack()

        self.master.bind('<Motion>', self.motion)

    def new_window(self, _class):
        self.new = Toplevel(self.master)
        _class(self.new)

        self.report_label = Label(master=self.new)

        #filter bboxes in classifying
        self.report_bboxes = {}
        self.new_bb = []
        for box in self.bboxes:
            crop_image = self.image.crop((box[0], box[1], box[2], box[3]))
            crop_image = crop_image.resize(size_image)
            dist, ind = self.classifying.predict_class(crop_image)
            if self.label_to_name[str(ind[0][0])] not in self.report_bboxes.keys() and dist < distance:
                self.report_bboxes[self.label_to_name[str(ind[0][0])]] = []
            if dist < distance:
                self.report_bboxes[self.label_to_name[str(ind[0][0])]].append(box)
                self.new_bb.append(box)

        #sort bboxes
        self.filter_bboxes = []
        for i in range(10):

            #find minbox
            minbox = [1024,1024,1024,1024]
            for box in self.new_bb:
                if box in np.array(self.filter_bboxes):
                    continue
                if minbox[0] + minbox[1] > box[0] + box[1]:
                    minbox = box
            self.filter_bboxes.append(minbox)

            #find near box
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
                    if ((lastbox[1] < box1[1] and lastbox[3] > box1[1]) or (lastbox[1] > box1[1] and lastbox[1] < box1[3])) and box1[0] - lastbox[2] < distance2box:
                        nearbox = box1
                        distance2box = box1[0] - lastbox[2]
                        s = 1
                if s == 1:
                    self.filter_bboxes.append(nearbox)

        #draw bboxes
        draw = ImageDraw.Draw(self.original_image)
        s = 1
        for key in self.report_bboxes:
            R = random.randint(0, 255)
            B = random.randint(0, 255)
            G = random.randint(0, 255)
            for box in self.report_bboxes[key]:
                draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline = (R,B,G), width = 2)
        for box in self.filter_bboxes:
            draw.text(((box[0] + box[2])/2, (box[1] + box[3])/2), text=str(s))
            s = s + 1
        self.showImg(self.report_label, self.original_image, 0, 0)

    def showImg(self,config, image, x, y):
        render = ImageTk.PhotoImage(image)  
        # labels can be text or images
        config.config(image = render)
        config.image = render
        config.place(x=x, y=y)

    def OpenFile(self):
        name = askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
                               filetypes=(( "Text File", "*.jpeg"), ("All Files", "*.*")),
                               title= "Choose a file."
                               )
        self.name = name
        return name

    def read_image(self):
        load = Image.open(self.OpenFile())

        self.image = load
        self.original_image = self.image.copy()
        return load

    def btn_open(self):
        render = self.read_image()
        self.showImg(self.store_img, render, 0, 0)

    def motion(self, event):
        if event.x > 1024 or event.y > 1024:
            return
        for box in self.bboxes:
            if event.x > box[0] and event.y > box[1] and event.x < box[2] and event.y < box[3]:
                crop_image = self.image.crop((box[0], box[1], box[2], box[3]))
                crop_image = crop_image.resize(size_image)
                dist, ind = self.classifying.predict_class(crop_image)

                name = self.label_to_name[str(ind[0][0])]
                image = Image.open(referencce_image_path + name)
                self.var.set(name + '\n' + str(dist[0][0]))
                self.showImg(self.reference_image, image, 1024, 0)
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
