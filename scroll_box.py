
import Tkinter
from PIL import Image, ImageTk
from sys import argv

import argparse
import numpy as np
import sys
from Featurizer import BBProposer, AlexNetFeaturizer

class BoxPicker:

    def __init__(self, image, outname):
        self.pix2boxes = {}
        self.feats = {}
        self.name = outname
        self.image = image
        self.clone = np.array(image)[:,:,:3].astype(np.float64)
        self.proposer = BBProposer()
        self.featurizer = AlexNetFeaturizer()
        self.clone[:,:] -= np.array([122.7717, 102.9801, 115.9465 ])
        boxes = [tuple(b) for b in self.proposer.extract_proposal(self.clone)]
        self.clone[:,:] += np.array([122.7717, 102.9801, 115.9465 ])
        crops = [self.proposer.get_crop(b, self.clone) for b in boxes]
        feats = {b: self.featurizer.getFeatures(c) for b,c in zip(boxes,crops)}

        for y in range(self.clone.shape[0]):
            for x in range(self.clone.shape[1]):
                self.pix2boxes[(x,y)] = []
        for b in boxes:
            x1, y1, x2, y2 = (int(bi) for bi in b)
            for y in range(y1, y2):
                for x in range(x1, x2):
                    self.pix2boxes[(x,y)].append(b)
            self.proposer.draw_box(b, self.clone, 0)
        window = Tkinter.Tk(className="Image")
        image = Image.fromarray(np.uint8(self.clone))
        canvas = Tkinter.Canvas(window, width=image.size[0], height=image.size[1])
        canvas.pack()
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvasimage = canvas.create_image(image.size[0]//2,
                                               image.size[1]//2,
                                               image=self.image_tk)
        self.feats = feats
        self.boxes = boxes
        canvas.bind("<Button-1>", self.click)
        self.canvas = canvas
        self.canvas.bind("<Right>", self.nextbox)
        self.canvas.bind("<Left>", self.prevbox)
        self.canvas.bind("<Return>", self.savefeats)
        self.canvas.focus_set()
        self.canvas.pack()

    def drawbox(self,box):
        self.proposer.draw_box(box, self.clone, 1)
        image = Image.fromarray(np.uint8(self.clone))
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.canvasimage, image=self.image_tk)

    def refreshboxes(self):
        for b in self.boxes:
            self.proposer.draw_box(b, self.clone, 0)
        image = Image.fromarray(np.uint8(self.clone))
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.canvasimage, image=self.image_tk)

    def nextbox(self, event):
        self.listpos+=1
        if self.listpos == len(self.boxlist):
            self.listpos = 0
        box = self.boxlist[self.listpos]
        self.refreshboxes()
        self.drawbox(box)

    def prevbox(self, event):
        self.listpos-=1
        if self.listpos == -1:
            self.listpos = len(self.boxlist)-1
        box = self.boxlist[self.listpos]
        self.refreshboxes()
        self.drawbox(box)

    def savefeats(self, event):
        print "Saving features to", self.name
        np.save(self.name, self.feats[self.boxlist[self.listpos]])

    def click(self,event):
        x,y = event.x, event.y
        self.refreshboxes()
        self.boxlist = self.pix2boxes[(x,y)]
        self.listpos = 0
        if len(self.boxlist) > 0:
            print "There are", len(self.boxlist), "boxes around this pixel."
            self.drawbox(self.boxlist[self.listpos])

        else:
            print "No box was selected, please try again"


image = Image.open(argv[1])
name = argv[2]
b = BoxPicker(image, name)
Tkinter.mainloop()
