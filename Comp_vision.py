import cv2
import numpy as np

class Comp_vision:


    def diff_backg(self, image, background):
        # es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,4))
        # background = cv2.imread('frames/bg.jpg')
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(background, im)
        diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
        # diff = cv2.dilate(diff, es, iterations = 2)
        img, counts, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
        for c in counts:
            (x, y, w, h) = cv2.boundingRect(c)
            x = (w/2 + x) + np.random.normal(0, 0.1, 1)*30
            y = (h/2 + y) + np.random.normal(0, 0.1, 1)*30

        return x[0], y[0] 

        # if cv2.waitKey(0) & 0xff == ord("q"):
        #     cv2.destroyAllWindows()

