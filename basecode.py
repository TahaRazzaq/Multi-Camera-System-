import os
import cv2
import time
import glob
import imutils
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = []
    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    for i in net.getUnconnectedOutLayers():
        output_layers.append(layer_names[i-1])
    return output_layers


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):
    global color_temp
    label = str(classes[class_id])

    color = COLORS[class_id]
    color_temp = color

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



def draw_bounding_box_base_pt(img, class_id, x, y, classes,color):
    
    label = str(classes[class_id])

    # color = COLORS[class_id]
    
    # print(color)

    cv2.circle(img, (round(x),round(y)), 1, color, thickness = 20)
    #cv2.ellipse(img, (round(x), round(y)), (50, 5), 0, 0, 360, color, -1)
    # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



def draw_bounding_box_base(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.circle(img, (round(x),round(y)), 1, color, thickness = 20)

    # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def toRGBA(img):

  ### Code Here ###
  if (img.dtype == np.uint8):
    img_new = img.astype(float) / 255.0
    # png = np.multiply(jpg, 1/255.0) # rgb values scaled between 0.0 to 1.0 floats from 0-255 uint8s
    alpha_row = np.ones((img.shape[0], img.shape[1], 1), dtype=float)
    rgba_dst = np.concatenate((img_new, alpha_row), axis=2)
  else:
    img_new = img.astype(float) / 255.0
    # png = np.multiply(jpg, 1/255.0) # rgb values scaled between 0.0 to 1.0 floats from 0-255 uint8s
    alpha_row = np.ones((img.shape[0], img.shape[1], 1), dtype=float)
    rgba_dst = np.concatenate((img_new, alpha_row), axis=2)

  return rgba_dst


def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

def detect(img1,cfg,weights,classes):
    width = img1.shape[1]
    height = img1.shape[0]

    scale = 0.00392

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    

    net = cv2.dnn.readNetFromDarknet(cfg, weights)


    blob = cv2.dnn.blobFromImage(img1, scale, (width,height), (0,0,0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))
    # print(outs)
    


    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.1
    nms_threshold = 0.7

    

    for out in outs:
        for detection in out:
            scores = detection[5:]
            
            class_idx = np.argmax(scores)
            # print(classval)
            confidence = scores[class_idx]
            # print(confidence)
            if confidence > 0.1:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                # class_idx = np.argmax(scores)
                if class_idx > (len(classes) - 1):
                    class_idx = class_idx % (len(classes)+1)
                # print(classval)
                class_ids.append(class_idx)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    return boxes, confidences, conf_threshold, nms_threshold, class_ids

    




def main():

    print("Press 1 for pre-recorded videos, 2 for live stream: ")
    option = int(input())

   
    # weights_pa1 = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/object-detection-opencv-master/object-detection-opencv-master/yolov3_mask_last.weights"
    # weights_pa1 = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/Task2/attempt_2/darknet/backup/mask-yolov3_10000.weights"
    # weights = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/object-detection-opencv-master/object-detection-opencv-master/yolov3.weights"
    classes = None

    cfg = "mask-yolov3.cfg"
        # cfg_2 = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/object-detection-opencv-master/object-detection-opencv-master/yolov3.cfg"
    classes_file = "mask-obj.names"
    classes_person = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/object-detection-opencv-master/object-detection-opencv-master/yolov3.txt"
    weights_person = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/object-detection-opencv-master/object-detection-opencv-master/yolov3.weights"
    cfg_person = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/object-detection-opencv-master/object-detection-opencv-master/yolov3.cfg"
    # classes_file = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/object-detection-opencv-master/object-detection-opencv-master/yolov3.txt"
    # weights = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/Task2/mask_3.weights"
    weights = "mask-yolov3_30000.weights"
    weights_2 = weights
    weights_3 = weights
    color_temp = 0
    colors = [(0,0,255),(255,0,255),(134,134,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]

    if option == 1:
        print("Press Y for Mask/No mask Detection\nPress P for Top View Projection\nPress d for Top View Projection and detection\nPress s for COVID SOP violation Projection\nPress h for Top View Projection and Heatmap")
        option_yolo = str(input())
        
        # weights_2 = "stream2_mask-yolov3_10000.weights"
        # weights_3 = "stream3_mask-yolov3_10000.weights"


        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        if option_yolo == "y":
            

            print("\nLoading images from 3 pre recorded feeds...")

            # img1 = cv2.imread("C:\\Users\\Hp\\OneDrive\\Documents\\University\\Semesters\\Fall 2021\\Computer Vision\\Mini_Project_1\\Yolo detector\\Yolo-Annotation-Tool-New--master_old\\Images\\stream1\\00000581.jpg")
            img1 = cv2.imread("00000160.jpg")
            img2 = cv2.imread("00000151.jpg")
            img3 = cv2.imread("00000287.jpg")

            # print(img1)

            # print(img2)

            # print(img3)

            width = img1.shape[1]
            height = img1.shape[0]

            width2 = img2.shape[1]
            height2 = img2.shape[0]

            width3 = img3.shape[1]
            height3 = img3.shape[0]

            scale = 0.00392

            COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
            

            net = cv2.dnn.readNetFromDarknet(cfg, weights)


            blob = cv2.dnn.blobFromImage(img1, scale, (width,height), (0,0,0), True, crop=False)
            # set input blob for the network
            net.setInput(blob)

            outs = net.forward(get_output_layers(net))
            # print(outs)
            


            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.1
            nms_threshold = 0.7

            

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    
                    class_idx = np.argmax(scores)
                    # print(classval)
                    confidence = scores[class_idx]
                    # print(confidence)
                    if confidence > 0.1:

                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        # class_idx = np.argmax(scores)
                        if class_idx > (len(classes) - 1):
                            class_idx = class_idx % (len(classes)+1)
                        # print(classval)
                        class_ids.append(class_idx)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])


            

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # print(class_ids)

            
            path = os.path.join(os.getcwd()+"/Detections/BB_prerecorded_S1.txt")
            f = open(path,'w')
            for i in indices:
                # i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                
                draw_bounding_box(img1, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")
                print("\nBounding boxes written to file...")



            # display output image    
            print("\nDisplaying Detection Results...")
            windowName_detect = "Detection Result"
            cv2.namedWindow(windowName_detect)
            cv2.imshow(windowName_detect, img1)



            


            net = cv2.dnn.readNetFromDarknet(cfg, weights_2)

            blob = cv2.dnn.blobFromImage(img2, scale, (width2,height2), (0,0,0), True, crop=False)
            # set input blob for the network
            net.setInput(blob)

            outs = net.forward(get_output_layers(net))
            # print(outs)
            


            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.1
            nms_threshold = 0.4

            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    
                    class_idx = np.argmax(scores)
                    # print(classval)
                    confidence = scores[class_idx]
                    # print(confidence)
                    if confidence > 0.1:

                        center_x = int(detection[0] * width2)
                        center_y = int(detection[1] * height2)
                        w = int(detection[2] * width2)
                        h = int(detection[3] * height2)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        # class_idx = np.argmax(scores)
                        if class_idx > (len(classes) - 1):
                            class_idx = class_idx % (len(classes)+1)
                            if class_idx > (len(classes) - 1):
                                class_idx = (len(classes) - 1)                        
                        # print(classval)
                        class_ids.append(class_idx)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])


            

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # print(class_ids)

            
            path = os.path.join(os.getcwd()+"/Detections/BB_prerecorded_S2.txt")
            f = open(path,'w')
            for i in indices:
                # i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                
                draw_bounding_box(img2, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")



            # display output image    
            windowName_detect_2 = "Detection Result 2"
            cv2.namedWindow(windowName_detect_2)
            cv2.imshow(windowName_detect_2, img2)


            # width3 = width2
            # height3 = height2


            net = cv2.dnn.readNetFromDarknet(cfg, weights_3)

            blob = cv2.dnn.blobFromImage(img3, scale, (width3,height3), (0,0,0), True, crop=False)
            # set input blob for the network
            net.setInput(blob)

            outs = net.forward(get_output_layers(net))
            # print(outs)
            


            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.1
            nms_threshold = 0.1

            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    
                    class_idx = np.argmax(scores)
                    # print(classval)
                    confidence = scores[class_idx]
                    # print(confidence)
                    if confidence > 0.1:

                        center_x = int(detection[0] * width3)
                        center_y = int(detection[1] * height3)
                        w = int(detection[2] * width3)
                        h = int(detection[3] * height3)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        # class_idx = np.argmax(scores)
                        # if class_idx > (len(classes) - 1):
                        #     class_idx = class_idx % (len(classes)+1)
                        # print(classval)
                        class_ids.append(class_idx)
                        confidences.append(float(confidence))
                        boxes.append([x, y, 1.5*w, 1.5*h])


            
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # print(class_ids)

            
            path = os.path.join(os.getcwd()+"/Detections/BB_prerecorded_S3.txt")
            f = open(path,'w')
            for i in indices:
                # i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                
                draw_bounding_box(img3, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")



            # display output image    
            windowName_detect_3 = "Detection Result 3"
            cv2.namedWindow(windowName_detect_3)
            cv2.imshow(windowName_detect_3, img3)

            print("\nPress any key to exit...")


            cv2.waitKey(0)

        elif option_yolo == "p":

            cap1 = cv2.VideoCapture('Recordings/Stream1Recording.avi')
            cap2 = cv2.VideoCapture('Recordings/Stream2Recording.avi')
            cap3 = cv2.VideoCapture('Recordings/Stream3Recording.avi')
            top_view = cv2.imread('Top_View.jpeg')
            size = (int(top_view.shape[1]),int(top_view.shape[0]))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            optputFile3 = cv2.VideoWriter(
                'TopView_PreRecorded_Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

            windowName = "Sample Feed from Camera 1"
            cv2.namedWindow(windowName)

            windowName2 = "Sample Feed from Camera 2"
            cv2.namedWindow(windowName2)

            windowName3 = "Sample Feed from Camera 3"
            cv2.namedWindow(windowName3)

            windowName4 = "Top View Of all Cameras"
            cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

            video = cv2.VideoWriter('./Recordings/TopView_Projections.avi', fourcc, 1, size)


            if (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                ret = True
            else:
                ret = False

            ind = 0
            images_array = []



            while (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                ret,img = cap1.read()
                ret1,img2 = cap2.read()
                ret2,img3 = cap3.read()
                top_view = cv2.imread('Top_View.jpeg')
                # img = cv2.imread("00000160.jpg")
                # img2 = cv2.imread("00000151.jpg")
                # img3 = cv2.imread("00000287.jpg")

                th,tw,_ = top_view.shape
                # print(th,tw)

                h,w,n = img3.shape
                # img3 = img3[377:h][0:w]
                # cv2.imwrite("cap_1.jpg",img)
                # cv2.imwrite("cap_2.jpg",img2)
                # cv2.imwrite("cap_3.jpg",img3)
                # break


            # img = cv2.imread('img_h.jpeg')
            # img2 = cv2.imread('img_t.jpeg')
            # # img = cv2.imread('00000287.jpg')
            # img3 = cv2.imread('img_0.jpeg')
            

                ipm_mat = []

                # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]
                i = 0
                # for pt in pts3:
                #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                #     i = i+1
                #     if i == len(colors):
                #         i = 0

                ## compute IPM matrix and apply it
                ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                ipm_mat.append(ipm1)
                cv2.imwrite('img1_transformed.jpg',ipm1)



                # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                ipm_mat.append(ipm2)
                # cv2.imwrite('img2_transformed.jpg',ipm2)
                # ipm_pts3 = np.array([[501,452],[655,452], [662,350], [507,350]], dtype=np.float32)
                # ipm_matrix = cv2.getPerspectiveTransform(pts3, ipm_pts3)
                # ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix, top_view.shape[:2][::-1])
                # ipm_mat.append(ipm3)


                # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                ipm_mat.append(ipm3)
                # cv2.imwrite('img3_transformed.jpg',ipm3)

                # print(ipm_mat)
                images_array = [img,img2,img3]

                # overlay = Image.open('img3_transformed.jpg').convert('RGBA')
                # image = Image.open('Top_View.jpeg').convert('RGBA')
                # image.paste(overlay, mask=overlay)
                # top_view_sol = image
                # image.save('result.png')




                h,w,n = top_view.shape
                top_view_sol = top_view.copy()
                top_view_sol = toRGBA(top_view_sol)
                # cv2.imshow("Top View Actual", top_view)
                # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                # (status, stitched) = stitcher.stitch(images_array)

                # cv2.imshow("stitched version", stitched)
                # ipm_mat = ipm_mat[0]

                for ipm in ipm_mat:
                    # ipm = toRGBA(ipm)
                    # print("Here")
                    for i in range(h):
                      for j in range(w):
                        location_px = ipm[i][j].copy()
                        if location_px[3] == 1:
                             # and (top_view_sol[i][j]>0.7).all())
                          #  and (location_px <0.1).any():
                          top_view_sol[i][j] = location_px


                im = Image.fromarray((top_view_sol * 255).astype(np.uint8))
                im.save("./Prerecorded_Video_Frames/Top_View"+str(ind)+".png")
                im = cv2.imread(("./Prerecorded_Video_Frames/Top_View"+str(ind)+".png"))
                video.write(im)
                ind = ind+1


                # plt.imshow(top_view_sol)
                # print(img.shape[:2][::-1])
                # cv2.imshow("top View",top_view_sol)
                # cv2.waitKey()

                # images_array.append(top_view_sol)
                # optputFile3.write(top_view_sol)

                # cv2.imwrite("Prerecorded_Video_Frames/TopView_PreRecorded_Recording_"+str(ind)+".png",top_view_sol)
                # ind = ind +1

                # display (or save) images
                cv2.imshow(windowName4, top_view_sol)
                cv2.imshow(windowName, img)
                cv2.imshow(windowName2, img2)
                cv2.imshow(windowName3, img3)
                # cv2.waitKey()
                # cv2.imshow('ipm', top_view_sol)
                if cv2.waitKey(1) == 27:
                    break

            # for imagee in images_array:
            #     optputFile3.write(imagee)
            cap1.release()
            # optputFile1.release()

            cap2.release()
            # optputFile2.release()

            cap3.release()
        # optputFile3.release()
            cv2.destroyAllWindows()
        elif option_yolo == "d":
            cap1 = cv2.VideoCapture('Recordings/Stream1Recording.avi')
            cap2 = cv2.VideoCapture('Recordings/Stream2Recording.avi')
            cap3 = cv2.VideoCapture('Recordings/Stream3Recording.avi')
            top_view = cv2.imread('Top_View.jpeg')
            size = (int(top_view.shape[1]),int(top_view.shape[0]))
            # optputFile3 = cv2.VideoWriter(
                # 'TopView_PreRecorded_Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

            windowName = "Sample Feed from Camera 1"
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

            windowName2 = "Sample Feed from Camera 2"
            cv2.namedWindow(windowName2,cv2.WINDOW_NORMAL)

            windowName3 = "Sample Feed from Camera 3"
            cv2.namedWindow(windowName3,cv2.WINDOW_NORMAL)

            windowName4 = "Top View Of all Cameras"
            cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter('./Recordings/TopView_Projections_Detections.avi', fourcc, 1, size)

            if (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                ret = True
            else:
                ret = False

            ind = 0
            images_array = []
            ind_bb = 0


            while (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                ret,img = cap1.read()
                ret1,img2 = cap2.read()
                ret2,img3 = cap3.read()
                top_view = cv2.imread('Top_View.jpeg')
                # img = cv2.imread("00000160.jpg")
                # img2 = cv2.imread("00000151.jpg")
                # img3 = cv2.imread("Stream1-2.jpg")
                # print(img3)

                th,tw,_ = top_view.shape
                
                # print(th,tw)

                h,w,n = img3.shape
                # img3 = img3[377:h][0:w]
                # cv2.imwrite("cap_1.jpg",img)
                # cv2.imwrite("cap_2.jpg",img2)
                # cv2.imwrite("cap_3.jpg",img3)
                # break


            # img = cv2.imread('img_h.jpeg')
            # img2 = cv2.imread('img_t.jpeg')
            # # img = cv2.imread('00000287.jpg')
            # img3 = cv2.imread('img_0.jpeg')
                # img = img2

                boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)


                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # print(class_ids)

            
                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                # COLORS = colors
                path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                f = open(path,'a')
                # print(indices)
                # print(boxes)
                # print(ipm_matrix3)
                circle_coord = []

                toDraw = []
                # color_array = []
                count = [0, 0, 0]
                # break
                img_detect = img.copy()
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    # print(ipm_matrix3*box)

                    # break
                    x = box[0]
                    # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    y =  box[1]
                    # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    w = box[2]
                    h = box[3]
                    
                    # print(x,y)
                    #draw_bounding_box_base(img, class_ids[i], confidences[i], round(x), round(y+200), round(x+w), round(y+h),classes,COLORS)
                    draw_bounding_box(img_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                    toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                    # color_array.append(COLORS[class_ids[i]])
                    count[0] = count[0] + 1
                    f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")

                    # cv2.circle(img, (round(x),round(y)), 10, COLORS[0], thickness=1, lineType=8, shift=0)
                    # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # print("\nDisplaying Detection Results...")
                # windowName_detect = "Detection Result"
                # cv2.namedWindow(windowName_detect)
                # cv2.imshow(windowName_detect, img_detect)




                boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img2, cfg, weights,classes)



                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # print(class_ids)

            
                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./Bounding_Boxes_Files/BB_prerecorded_S2"+str(ind_bb)+".txt"
                f = open(path,'a')
                # print(indices)
                # print(boxes)
                # print(ipm_matrix3)
                circle_coord = []

                # break
                img2_detect = img2.copy()
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    # print(ipm_matrix3*box)

                    # break
                    x = box[0]
                    # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    y =  box[1]
                    # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    w = box[2]
                    h = box[3]
                    
                    # print(x,y)
                    #draw_bounding_box_base(img2, class_ids[i], confidences[i], round(x), round(y+170), round(x+w), round(y+h),classes,COLORS)
                    draw_bounding_box(img2_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                    toDraw.append([round(x), round(y+175), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                    # print(colors[class_ids[i]])
                    count[1] = count[1] + 1
                    f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")







                
                boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img3, cfg, weights,classes)


                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # print(class_ids)

            
                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./Bounding_Boxes_Files/BB_prerecorded_S3"+str(ind_bb)+".txt"
                f = open(path,'a')
                # print(indices)
                # print(boxes)
                # print(ipm_matrix3)
                circle_coord = []

                # break
                img3_detect = img3.copy()
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    # print(ipm_matrix3*box)

                    # break
                    x = box[0]
                    # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    y =  box[1]
                    # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    w = box[2]
                    h = box[3]
                    
                    # print(x,y)
                    #draw_bounding_box_base(img3, class_ids[i], confidences[i], round(x), round(y+120), round(x+w), round(y+h),classes,COLORS)
                    draw_bounding_box(img3_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                    toDraw.append([round(x), round(y+120), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                    count[2] = count[2] + 1
                    f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")


                ipm_mat = []
                ind_bb = ind_bb +1

                # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                
                # i = 0
                # for pt in pts3:
                #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                #     i = i+1
                #     if i == len(colors):
                #         i = 0

                ## compute IPM matrix and apply it
                ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                ipm_mat.append(ipm1)
                # cv2.imwrite('img1_transformed.jpg',ipm1)



                # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                ipm_mat.append(ipm2)
                # cv2.imwrite('img2_transformed.jpg',ipm2)
                # ipm_pts3 = np.array([[501,452],[655,452], [662,350], [507,350]], dtype=np.float32)
                # ipm_matrix = cv2.getPerspectiveTransform(pts3, ipm_pts3)
                # ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix, top_view.shape[:2][::-1])
                # ipm_mat.append(ipm3)


                # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                ipm_mat.append(ipm3)
                # cv2.imwrite('img3_transformed.jpg',ipm3)


                toDrawPts = []
                for i in range(0, count[0]):
                    #xpt = (toDraw[i][0] + toDraw[i][2])/2
                    #ypt = (toDraw[i][1] + toDraw[i][3])/2
                    #inter = np.matmul(ipm_matrix, np.array([round(xpt), round(ypt), 1]))
                    inter = np.matmul(ipm_matrix, np.array([toDraw[i][0], toDraw[i][1], 1]))
                    inter[0] = inter[0]/inter[2]
                    inter[1] = inter[1]/inter[2]
                    inter[2] = 1
                    toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                for i in range(count[0], count[0]+count[1]):
                    #xpt = (toDraw[i][0] + toDraw[i][2])/2
                    #ypt = (toDraw[i][1] + toDraw[i][3])/2
                    #inter = np.matmul(ipm_matrix2, np.array([round(xpt), round(ypt), 1]))
                    inter = np.matmul(ipm_matrix2, np.array([toDraw[i][0], toDraw[i][1], 1]))
                    inter[0] = inter[0]/inter[2]
                    inter[1] = inter[1]/inter[2]
                    inter[2] = 1
                    toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                for i in range(count[0]+count[1], count[0]+count[1]+count[2]):
                    #xpt = (toDraw[i][0] + toDraw[i][2])/2
                    #ypt = (toDraw[i][1] + toDraw[i][3])/2
                    #inter = np.matmul(ipm_matrix3, np.array([round(xpt), round(ypt), 1]))
                    inter = np.matmul(ipm_matrix3, np.array([toDraw[i][0], toDraw[i][1], 1]))
                    inter[0] = inter[0]/inter[2]
                    inter[1] = inter[1]/inter[2]
                    inter[2] = 1
                    toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])


                #print("SHAPE", toDrawPts[0].shape, toDrawPts[0][0], toDrawPts[0][1], toDrawPts[0][2])
                # ipm_mat.append(ipm3)




                # print(ipm_mat)
                images_array = [img,img2,img3]

                # overlay = Image.open('img3_transformed.jpg').convert('RGBA')
                # image = Image.open('Top_View.jpeg').convert('RGBA')
                # image.paste(overlay, mask=overlay)
                # top_view_sol = image
                # image.save('result.png')




                h,w,n = top_view.shape
                top_view_sol = top_view.copy()
                top_view_sol = toRGBA(top_view_sol)
                # cv2.imshow("Top View Actual", top_view)
                # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                # (status, stitched) = stitcher.stitch(images_array)

                # cv2.imshow("stitched version", stitched)
                # ipm_mat = ipm_mat[0]

                # print(circle_coord[0][0])

                for ipm in ipm_mat:
                    # ipm = toRGBA(ipm)
                    # print("Here")
                    for i in range(h):
                      for j in range(w):
                        location_px = ipm[i][j].copy()
                        if location_px[3] == 1:
                             # and (top_view_sol[i][j]>0.7).all())
                          #  and (location_px <0.1).any():
                          top_view_sol[i][j] = location_px


                # print(color_temp)
                for i in range(len(toDrawPts)):
                    draw_bounding_box_base_pt(top_view_sol, toDrawPts[i][2], toDrawPts[i][0], toDrawPts[i][1], toDrawPts[i][3], toDrawPts[i][4])

                

                im = Image.fromarray((top_view_sol * 255).astype(np.uint8))
                im.save("./Prerecorded_Video_frames_Detection/Top_View"+str(ind)+".png")
                im = cv2.imread(("./Prerecorded_Video_frames_Detection/Top_View"+str(ind)+".png"))
                video.write(im)
                ind = ind+1
                # break`


                # top_view_sol = cv2.cvtColor(top_view_sol, cv2.COLOR_RGBA2RGB)

                # rgbImage = cv2.cvtColor(npimage, cv2.COLOR_RGBA2RGB)


                # cv2.circle(top_view_sol, (int(circle_coord[0][0][0]),int(circle_coord[0][0][1])), 10, COLORS[0], thickness=10, lineType=8, shift=0)


                # plt.imshow(top_view_sol)
                # print(img.shape[:2][::-1])
                # cv2.imshow("top View",top_view_sol)
                # cv2.waitKey()

                # images_array.append(top_view_sol)
                # optputFile3.write(top_view_sol)

                # cv2.imwrite("Prerecorded_Video_Frames/TopView_PreRecorded_Recording_"+str(ind)+".png",top_view_sol)
                # ind = ind +1

                # display (or save) images
                cv2.imshow(windowName4, top_view_sol)
                cv2.imshow(windowName, img_detect)
                cv2.imshow(windowName2, img2_detect)
                cv2.imshow(windowName3, img3_detect)
                # cv2.waitKey()
                # cv2.imshow('ipm', top_view_sol)
                if cv2.waitKey(1) == 27:
                    break

            # for imagee in images_array:
            #     optputFile3.write(imagee)
            
            

            imgs_list = glob.glob("./Prerecorded_Video_Frames/*.png")
            for imgs in imgs_list:
                img = cv2.imread(imgs)
                video.write(img)

              

            cap1.release()
            # optputFile1.release()

            cap2.release()
            # optputFile2.release()

            cap3.release()
        # optputFile3.release()
            cv2.destroyAllWindows()

        elif option_yolo == "s":
            cap1 = cv2.VideoCapture('Recordings/Stream1Recording.avi')
            cap2 = cv2.VideoCapture('Recordings/Stream2Recording.avi')
            cap3 = cv2.VideoCapture('Recordings/Stream3Recording.avi')
            top_view = cv2.imread('Top_View.jpeg')
            size = (int(top_view.shape[1]),int(top_view.shape[0]))
            # optputFile3 = cv2.VideoWriter(
                # 'TopView_PreRecorded_Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

            windowName = "Sample Feed from Camera 1"
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

            windowName2 = "Sample Feed from Camera 2"
            cv2.namedWindow(windowName2,cv2.WINDOW_NORMAL)

            windowName3 = "Sample Feed from Camera 3"
            cv2.namedWindow(windowName3,cv2.WINDOW_NORMAL)

            windowName4 = "Top View Of all Cameras"
            cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter('./Recordings/COVID_SOP_TopView.avi', fourcc, 1, size)

            if (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                ret = True
            else:
                ret = False

            ind = 0
            images_array = []
            ind_bb = 0

            px_img1 = (((1099-863) *2)/1.245)

            px_img2 = (((357-198) *2)/1.245)

            px_img3 = (((1297-1162) *2)/1.245)
            print(px_img1)
            print(px_img2)
            print(px_img3)
            colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]

            # cfg_person = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/darknet/cfg/yolov3.cfg"
            # weights_person = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/darknet/yolov3.weights"
            classes_person_file = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/darknet/cfg/coco.data"

            with open(classes_person_file, 'r') as f:
                classes_person = [line.strip() for line in f.readlines()]




            while (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                ret,img = cap1.read()
                ret1,img2 = cap2.read()
                ret2,img3 = cap3.read()
                top_view = cv2.imread('Top_View.jpeg')
                # img = cv2.imread("00000160.jpg")
                # img2 = cv2.imread("00000151.jpg")
                # img3 = cv2.imread("Stream1-2.jpg")
                # print(img3)

                th,tw,_ = top_view.shape
                # print(th,tw)

                h,w,n = img3.shape
                # img3 = img3[377:h][0:w]
                # cv2.imwrite("cap_1.jpg",img)
                # cv2.imwrite("cap_2.jpg",img2)
                # cv2.imwrite("cap_3.jpg",img3)
                # break


            # img = cv2.imread('img_h.jpeg')
            # img2 = cv2.imread('img_t.jpeg')
            # # img = cv2.imread('00000287.jpg')
            # img3 = cv2.imread('img_0.jpeg')
                # img = img2



                # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)

                width = img.shape[1]
                height = img.shape[0]

                scale = 0.00392

                COLORS = np.random.uniform(0, 255, size=(len(classes_person), 3))
                # COLORS = colors
                

                net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                layer_names = net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
                # set input blob for the network
                net.setInput(blob)

                outs = net.forward(output_layers)
                # print(outs)
                


                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.7

                

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        
                        class_idx = np.argmax(scores)
                        # print(classval)
                        confidence = scores[class_idx]
                        # print(confidence)
                        if confidence > 0.5:

                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            # class_idx = np.argmax(scores)
                            if class_idx > (len(classes) - 1):
                                class_idx = class_idx % (len(classes)+1)
                            # print(classval)
                            if class_idx == 0:
                                class_ids.append(class_idx)
                                confidences.append(float(confidence))
                                boxes.append([x, y, w, h])

                # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # print(class_ids)

                classes_covid = ["violation","no_violation"]
                # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                f = open(path,'a')
                # print(indices)
                # print(boxes)
                # print(ipm_matrix3)
                circle_coord = []
                toDraw = []
                # color_array = []
                count = [0, 0, 0]

                # break
                img_detect = img.copy()
                box = [boxes[i] for i in indices]
                # print(box)
                class_id_covid = []
                # if len(box) == 1:


                for i in range(len(box)-1):
                    if i + 1 != len(box):
                        dist = box[i][0] - box[i+1][0]
                        if dist > px_img1:
                            class_id_covid.append(1)
                        else:
                            class_id_covid.append(0)
                        x = box[i][0]
                        # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[i][1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = (box[i][2] + box[i+1][2])
                        h = (box[i][3])
                    else:
                        class_id_covid.append(1)

                    

                    # x = box[i][0]
                    # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    # y =  box[i][1]
                    # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    # w = box[i][2]
                    # h = box[i][3]
                    # draw_bounding_box_base(img, class_id_covid[i], confidences[i], round(x+100), round(y+200), round(x+w), round(y+h),classes_covid,COLORS)
                    toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                    # color_array.append(COLORS[class_ids[i]])
                    count[0] = count[0] + 1
                    draw_bounding_box(img_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+(w)), round(y+(h)),classes_covid,COLORS)



                net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                layer_names = net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                blob = cv2.dnn.blobFromImage(img2, scale, (416,416), (0,0,0), True, crop=False)
                # set input blob for the network
                net.setInput(blob)

                outs = net.forward(output_layers)
                # print(outs)
                


                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.95
                nms_threshold = 0.7

                

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        
                        class_idx = np.argmax(scores)
                        # print(class_idx)
                        # print(classval)
                        confidence = scores[class_idx]
                        # print(confidence)
                        if confidence > 0.5:

                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            # class_idx = np.argmax(scores)
                            if class_idx > (len(classes) - 1):
                                class_idx = class_idx % (len(classes)+1)
                            # print(classval)
                            if class_idx == 0:
                                class_ids.append(class_idx)
                                confidences.append(float(confidence))
                                boxes.append([x, y, w, h])


                # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img2, cfg, weights,classes)
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # print(class_ids)

                classes_covid = ["violation","no_violation"]
                # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                f = open(path,'a')
                # print(indices)
                # print(boxes)
                # print(ipm_matrix3)
                circle_coord = []

                # break
                img2_detect = img2.copy()
                box = [boxes[i] for i in indices]
                # print(box)
                class_id_covid = []
                # if len(box) == 1:


                for i in range(len(box)-1):
                    if i + 1 != len(box):
                        dist = box[i][0] - box[i+1][0]
                        if dist > px_img2:
                            class_id_covid.append(1)
                        else:
                            class_id_covid.append(0)
                        x = box[i][0]
                        # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[i][1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = box[i][2] + box[i+1][2]
                        h = box[i][3] 
                        # h = h + 50
                        # y = y + 50
                        # x = x+ 50
                    else:
                        class_id_covid.append(1)

                    

                    # x = box[i][0]
                    # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    # y =  box[i][1]
                    # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    # w = box[i][2]
                    # h = box[i][3]
                    # draw_bounding_box_base(img2, class_id_covid[i], confidences[i], round(x+130), round(y+200), round(x+w), round(y+h),classes_covid,COLORS)
                    toDraw.append([round(x+130), round(y+200), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                    # color_array.append(COLORS[class_ids[i]])
                    count[1] = count[1] + 1
                    draw_bounding_box(img2_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes_covid,COLORS)


                # net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                # layer_names = net.getLayerNames()
                # output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                blob = cv2.dnn.blobFromImage(img3, scale, (416,416), (0,0,0), True, crop=False)
                # set input blob for the network
                net.setInput(blob)

                outs = net.forward(output_layers)
                # print(outs)
                


                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.7

                

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        
                        class_idx = np.argmax(scores)
                        # print(classval)
                        confidence = scores[class_idx]
                        # print(confidence)
                        if confidence > 0.5:

                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            # class_idx = np.argmax(scores)
                            if class_idx > (len(classes) - 1):
                                class_idx = class_idx % (len(classes)+1)
                            # print(classval)
                            if class_idx == 0:
                                class_ids.append(class_idx)
                                confidences.append(float(confidence))
                                boxes.append([x, y, w, h])


                # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img3, cfg, weights,classes)
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # print(class_ids)


                classes_covid = ["violation","no_violation"]
                # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                f = open(path,'a')
                # print(indices)
                # print(boxes)
                # print(ipm_matrix3)
                circle_coord = []

                # break
                img3_detect = img3.copy()
                box = [boxes[i] for i in indices]
                # print(box)
                # print("confidence::", confidences)
                class_id_covid = []
                # if len(box) == 1:


                for i in range(len(box)-1):
                    if i + 1 != len(box):
                        dist = box[i][0] - box[i+1][0]
                        if dist > px_img2:
                            class_id_covid.append(1)
                        else:
                            class_id_covid.append(0)
                        x = box[i][0]
                        # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[i][1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = box[i][2] + box[i+1][2]
                        h = box[i][3]
                    else:
                        class_id_covid.append(1)

                    

                    # x = box[i][0]
                    # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    # y =  box[i][1]
                    # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    # w = box[i][2]
                    # h = box[i][3]
                    # draw_bounding_box_base(img3, class_id_covid[i], confidences[i], round(x+50), round(y+180), round(x+w), round(y+h),classes_covid,COLORS)
                    toDraw.append([ round(x+50), round(y+180), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                    # color_array.append(COLORS[class_ids[i]])
                    count[2] = count[2] + 1
                    draw_bounding_box(img3_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes_covid,COLORS)
                


                ipm_mat = []
                ind_bb = ind_bb +1

                # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]
                # i = 0
                # for pt in pts3:
                #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                #     i = i+1
                #     if i == len(colors):
                #         i = 0

                ## compute IPM matrix and apply it
                ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                ipm_mat.append(ipm1)
                # cv2.imwrite('img1_transformed.jpg',ipm1)



                # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                ipm_mat.append(ipm2)
              


                # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                ipm_mat.append(ipm3)
                # cv2.imwrite('img3_transformed.jpg',ipm3)


                
                # ipm_mat.append(ipm3)


                toDrawPts = []
                for i in range(0, count[0]):
                    #xpt = (toDraw[i][0] + toDraw[i][2])/2
                    #ypt = (toDraw[i][1] + toDraw[i][3])/2
                    #inter = np.matmul(ipm_matrix, np.array([round(xpt), round(ypt), 1]))
                    inter = np.matmul(ipm_matrix, np.array([toDraw[i][0], toDraw[i][1], 1]))
                    inter[0] = inter[0]/inter[2]
                    inter[1] = inter[1]/inter[2]
                    inter[2] = 1
                    toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                for i in range(count[0], count[0]+count[1]):
                    #xpt = (toDraw[i][0] + toDraw[i][2])/2
                    #ypt = (toDraw[i][1] + toDraw[i][3])/2
                    #inter = np.matmul(ipm_matrix2, np.array([round(xpt), round(ypt), 1]))
                    inter = np.matmul(ipm_matrix2, np.array([toDraw[i][0], toDraw[i][1], 1]))
                    inter[0] = inter[0]/inter[2]
                    inter[1] = inter[1]/inter[2]
                    inter[2] = 1
                    toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                for i in range(count[0]+count[1], count[0]+count[1]+count[2]):
                    #xpt = (toDraw[i][0] + toDraw[i][2])/2
                    #ypt = (toDraw[i][1] + toDraw[i][3])/2
                    #inter = np.matmul(ipm_matrix3, np.array([round(xpt), round(ypt), 1]))
                    inter = np.matmul(ipm_matrix3, np.array([toDraw[i][0], toDraw[i][1], 1]))
                    inter[0] = inter[0]/inter[2]
                    inter[1] = inter[1]/inter[2]
                    inter[2] = 1
                    toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])


                #print("SHAPE", toDrawPts[0].shape, toDrawPts[0][0], toDrawPts[0][1], toDrawPts[0][2])




                # print(ipm_mat)
                images_array = [img,img2,img3]

           




                h,w,n = top_view.shape
                top_view_sol = top_view.copy()
                top_view_sol = toRGBA(top_view_sol)
                top_view_sol_copy = top_view_sol.copy()
                # cv2.imshow("Top View Actual", top_view)
                # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                # (status, stitched) = stitcher.stitch(images_array)

                # cv2.imshow("stitched version", stitched)
                # ipm_mat = ipm_mat[0]

                # print(circle_coord[0][0])

                for ipm in ipm_mat:
                    # ipm = toRGBA(ipm)
                    # print("Here")
                    for i in range(h):
                      for j in range(w):
                        location_px = ipm[i][j].copy()
                        if location_px[3] == 1:
                             # and (top_view_sol[i][j]>0.7).all())
                          #  and (location_px <0.1).any():
                          top_view_sol[i][j] = location_px



                for i in range(len(toDrawPts)):
                    draw_bounding_box_base_pt(top_view_sol, toDrawPts[i][2], toDrawPts[i][0], toDrawPts[i][1], toDrawPts[i][3], toDrawPts[i][4])

                


                im = Image.fromarray((top_view_sol * 255).astype(np.uint8))
                im.save("./Prerecorded_Video_frames_Covid/Top_View"+str(ind)+".png")
                im = cv2.imread(("./Prerecorded_Video_frames_Covid/Top_View"+str(ind)+".png"))
                video.write(im)
                ind = ind+1
                


               
                
                # display (or save) images
                cv2.imshow(windowName, img_detect)
                cv2.imshow(windowName2, img2_detect)
                cv2.imshow(windowName3, img3_detect)
                cv2.imshow(windowName4, top_view_sol)
                # cv2.waitKey()
                # cv2.imshow('ipm', top_view_sol)
                if cv2.waitKey(1) == 27:
                    break

            # for imagee in images_array:
            #     optputFile3.write(imagee)
            
            

            imgs_list = glob.glob("./Prerecorded_Video_Frames/*.png")
            for imgs in imgs_list:
                img = cv2.imread(imgs)
                video.write(img)

              

            cap1.release()
            # optputFile1.release()

            cap2.release()
            # optputFile2.release()

            cap3.release()
        # optputFile3.release()
            cv2.destroyAllWindows()

        elif option_yolo == "h":

            print("Press s for static heatmap\nPress a for animated heatmap \nPress c for COVID SOP Violated heatmap")
            option_heat = str(input())

            if option_heat == "s":
               


                cap1 = cv2.VideoCapture('Recordings/Stream1Recording.avi')
                cap2 = cv2.VideoCapture('Recordings/Stream2Recording.avi')
                cap3 = cv2.VideoCapture('Recordings/Stream3Recording.avi')
                top_view = cv2.imread('Top_View.jpeg')
                size = (int(top_view.shape[1]),int(top_view.shape[0]))
                # optputFile3 = cv2.VideoWriter(
                    # 'TopView_PreRecorded_Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

                windowName = "Sample Feed from Camera 1"
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

                windowName2 = "Sample Feed from Camera 2"
                cv2.namedWindow(windowName2,cv2.WINDOW_NORMAL)

                windowName3 = "Sample Feed from Camera 3"
                cv2.namedWindow(windowName3,cv2.WINDOW_NORMAL)

                windowName4 = "Top View Of all Cameras"
                cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter('./Recordings/Static_heatmap_PreRecorded.avi', fourcc, 1, size)

                if (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                    ret = True
                else:
                    ret = False

                ind = 0
                images_array = []
                heatmap_global = None
                ind_bb = 0


                while (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                    ret,img = cap1.read()
                    ret1,img2 = cap2.read()
                    ret2,img3 = cap3.read()
                    top_view = cv2.imread('Top_View.jpeg')
                    # img = cv2.imread("00000160.jpg")
                    # img2 = cv2.imread("00000151.jpg")
                    # img3 = cv2.imread("Stream1-2.jpg")
                    # print(img3)

                    th,tw,_ = top_view.shape
                    # print(th,tw)

                    h,w,n = img3.shape
                    # img3 = img3[377:h][0:w]
                    # cv2.imwrite("cap_1.jpg",img)
                    # cv2.imwrite("cap_2.jpg",img2)
                    # cv2.imwrite("cap_3.jpg",img3)
                    # break


                # img = cv2.imread('img_h.jpeg')
                # img2 = cv2.imread('img_t.jpeg')
                # # img = cv2.imread('00000287.jpg')
                # img3 = cv2.imread('img_0.jpeg')
                    # img = img2

                    boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)


                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                
                    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                    path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                    f = open(path,'a')
                    # print(indices)
                    # print(boxes)
                    # print(ipm_matrix3)
                    circle_coord = []

                    toDraw = []
                    count = [0, 0, 0]
                    # break
                    img_detect = img.copy()
                    for i in indices:
                        # i = i[0]
                        box = boxes[i]
                        # print(ipm_matrix3*box)

                        # break
                        x = box[0]
                        # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = box[2]
                        h = box[3]
                        
                        # print(x,y)
                        #draw_bounding_box_base(img, class_ids[i], confidences[i], round(x), round(y+200), round(x+w), round(y+h),classes,COLORS)
                        toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                        count[0] = count[0] + 1
                        draw_bounding_box(img_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                        f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")

                        # cv2.circle(img, (round(x),round(y)), 10, COLORS[0], thickness=1, lineType=8, shift=0)
                        # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # print("\nDisplaying Detection Results...")
                    # windowName_detect = "Detection Result"
                    # cv2.namedWindow(windowName_detect)
                    # cv2.imshow(windowName_detect, img_detect)




                    boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img2, cfg, weights,classes)


                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                
                    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                    path = "./Bounding_Boxes_Files/BB_prerecorded_S2"+str(ind_bb)+".txt"
                    f = open(path,'a')
                    # print(indices)
                    # print(boxes)
                    # print(ipm_matrix3)
                    circle_coord = []

                    # break
                    img2_detect = img2.copy()
                    for i in indices:
                        # i = i[0]
                        box = boxes[i]
                        # print(ipm_matrix3*box)

                        # break
                        x = box[0]
                        # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = box[2]
                        h = box[3]
                        
                        # print(x,y)
                        #draw_bounding_box_base(img2, class_ids[i], confidences[i], round(x), round(y+170), round(x+w), round(y+h),classes,COLORS)
                        toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                        count[1] = count[1] + 1
                        draw_bounding_box(img2_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                        f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")







                    
                    boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img3, cfg, weights,classes)


                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                
                    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                    path = "./Bounding_Boxes_Files/BB_prerecorded_S3"+str(ind_bb)+".txt"
                    f = open(path,'a')
                    # print(indices)
                    # print(boxes)
                    # print(ipm_matrix3)
                    circle_coord = []

                    # break
                    img3_detect = img3.copy()
                    for i in indices:
                        # i = i[0]
                        box = boxes[i]
                        # print(ipm_matrix3*box)

                        # break
                        x = box[0]
                        # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = box[2]
                        h = box[3]
                        
                        # print(x,y)
                        #draw_bounding_box_base(img3, class_ids[i], confidences[i], round(x), round(y+120), round(x+w), round(y+h),classes,COLORS)
                        toDraw.append([round(x), round(y+120), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                        count[2] = count[2] + 1
                        draw_bounding_box(img3_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                        f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")


                    ipm_mat = []
                    ind_bb = ind_bb +1

                    # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                    # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                    # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                    # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                    # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                    pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                    pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                    pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                    pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                    # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                    pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                    # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                    # colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]
                    # i = 0
                    # for pt in pts3:
                    #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                    #     i = i+1
                    #     if i == len(colors):
                    #         i = 0

                    ## compute IPM matrix and apply it
                    ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                    # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                    # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                    ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                    # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                    ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                    # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                    ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                    ipm_mat.append(ipm1)
                    # cv2.imwrite('img1_transformed.jpg',ipm1)



                    # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                    # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                    ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                    ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                    # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                    ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                    ipm_mat.append(ipm2)
                    # cv2.imwrite('img2_transformed.jpg',ipm2)
                    # ipm_pts3 = np.array([[501,452],[655,452], [662,350], [507,350]], dtype=np.float32)
                    # ipm_matrix = cv2.getPerspectiveTransform(pts3, ipm_pts3)
                    # ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix, top_view.shape[:2][::-1])
                    # ipm_mat.append(ipm3)


                    # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                    ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                    # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                    ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                    # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                    ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                    ipm_mat.append(ipm3)
                    # cv2.imwrite('img3_transformed.jpg',ipm3)


                    toDrawPts = []
                    for i in range(0, count[0]):
                        #xpt = (toDraw[i][0] + toDraw[i][2])/2
                        #ypt = (toDraw[i][1] + toDraw[i][3])/2
                        #inter = np.matmul(ipm_matrix, np.array([round(xpt), round(ypt), 1]))
                        inter = np.matmul(ipm_matrix, np.array([toDraw[i][0], toDraw[i][1], 1]))
                        inter[0] = inter[0]/inter[2]
                        inter[1] = inter[1]/inter[2]
                        inter[2] = 1
                        toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                    for i in range(count[0], count[0]+count[1]):
                        #xpt = (toDraw[i][0] + toDraw[i][2])/2
                        #ypt = (toDraw[i][1] + toDraw[i][3])/2
                        #inter = np.matmul(ipm_matrix2, np.array([round(xpt), round(ypt), 1]))
                        inter = np.matmul(ipm_matrix2, np.array([toDraw[i][0], toDraw[i][1], 1]))
                        inter[0] = inter[0]/inter[2]
                        inter[1] = inter[1]/inter[2]
                        inter[2] = 1
                        toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                    for i in range(count[0]+count[1], count[0]+count[1]+count[2]):
                        #xpt = (toDraw[i][0] + toDraw[i][2])/2
                        #ypt = (toDraw[i][1] + toDraw[i][3])/2
                        #inter = np.matmul(ipm_matrix3, np.array([round(xpt), round(ypt), 1]))
                        inter = np.matmul(ipm_matrix3, np.array([toDraw[i][0], toDraw[i][1], 1]))
                        inter[0] = inter[0]/inter[2]
                        inter[1] = inter[1]/inter[2]
                        inter[2] = 1
                        toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])


                    #print("SHAPE", toDrawPts[0].shape, toDrawPts[0][0], toDrawPts[0][1], toDrawPts[0][2])
                    # ipm_mat.append(ipm3)




                    # print(ipm_mat)
                    images_array = [img,img2,img3]

                    # overlay = Image.open('img3_transformed.jpg').convert('RGBA')
                    # image = Image.open('Top_View.jpeg').convert('RGBA')
                    # image.paste(overlay, mask=overlay)
                    # top_view_sol = image
                    # image.save('result.png')




                    top_view = cv2.cvtColor(cv2.imread('Top_View.jpeg'),cv2.COLOR_BGR2RGB)/255
                    h,w,n = top_view.shape
                    top_view_sol = top_view.copy()
                    top_view_sol = toRGBA(top_view_sol)
                    # cv2.imshow("Top View Actual", top_view)
                    # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                    # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                    # (status, stitched) = stitcher.stitch(images_array)

                    # cv2.imshow("stitched version", stitched)
                    # ipm_mat = ipm_mat[0]

                    # print(circle_coord[0][0])

                    for ipm in ipm_mat:
                        # ipm = toRGBA(ipm)
                        # print("Here")
                        for i in range(h):
                          for j in range(w):
                            location_px = ipm[i][j].copy()
                            if location_px[3] == 1:
                                 # and (top_view_sol[i][j]>0.7).all())
                              #  and (location_px <0.1).any():
                              top_view_sol[i][j] = location_px


                    for i in range(len(toDrawPts)):
                        draw_bounding_box_base_pt(top_view_sol, toDrawPts[i][2], toDrawPts[i][0], toDrawPts[i][1], toDrawPts[i][3], toDrawPts[i][4])

                                   # cv2.cvtColor(top_view_sol,cv2.COLOR_BGR2RGB)/255


                    


                    k = 21
                        # x=0
                    gauss = cv2.getGaussianKernel(k,np.sqrt(64))
                    gauss = gauss * gauss.T
                    gauss = (gauss/gauss[int(k/2),int(k/2)])

                    heatmap = np.zeros((top_view.shape[0],top_view.shape[1],3)).astype(np.float32)
                    # heatmap = toRGBA(heatmap)
                    # heatmap[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])] += 1
                    # heatmap[heatmap > 0.8] = 0


                    j = cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),cv2.COLORMAP_SUMMER),cv2.COLOR_BGR2RGB).astype(np.float32)/255
                    points = toDrawPts

                    for p in points:
                        # print(p[0]- int(k/2))
                        # print(p[0]+int(k/2)+1)
                        # print(p[1]- int(k/2))
                        # print(p[1]+int(k/2)+1)

                        # print(heatmap[int(p[0]):int(p[0]+int(k/2)+1),int(p[1]-int(k/2)):int(p[1]+int(k/2)+1),:].shape)
                        # print(j.shape)



                        try:
                            # b = heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:]
                            # c = j + b

                            # heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:] = j
                            draw_bounding_box_base_pt(heatmap, p[2], p[0], p[1], p[3], p[4])
                        except:
                            pass

                    # m = np.max(heatmap, axis = (0,1))
                    # heatmap = heatmap/m




                    heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)

                    mask = np.where(heatmap > 0.2,1,0).astype(np.float32)
                    inv_mask = np.ones((top_view.shape[0],top_view.shape[1],3))*(1 - mask)[:,:,None]
                    
                    im = Image.fromarray((heatmap * 255).astype(np.uint8))
                    im.save('./frames_S1/frames'+str(ind)+".png")


                    images = cv2.imread('./frames_S1/frames'+str(ind)+".png", 0)
                    # colormap = plt.get_cmap('inferno')
                    # heatmap = (colormap(images) * 2**16).astype(np.uint16)[:,:,:3]
                    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

                    # heatmap = (heatmap/255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(images, cv2.COLORMAP_AUTUMN)
                    # try:
                    #     heatmap_global == None:
                    #     heatmap_global = heatmap
                    # else:
                    try:
                        heatmap_global = heatmap_global + heatmap
                    except:
                        heatmap_global = heatmap
                    # heatmap_array.append(heatmap)
                    # # imask = (inv_mask/255).astype(np.uint8)
                    # # imask = cv2.applyColorMap(imask, cv2.COLORMAP_HOT)

                    
                    new_top = np.where(inv_mask>0,top_view,inv_mask)
                    new_top = inv_mask * top_view

                    new_top = new_top + heatmap_global





                    # cv2.imshow("New_Top_View",new_top)
                    # cv2.imshow("Heatmap",heatmap)
                    # cv2.imshow("Top View Sol",top_view_sol)
                    # cv2.imshow("Image",img_detect)



                    # cv2.imshow("Actual img",img)

                    # cv2.waitKey()
                    

                    

                    im = Image.fromarray((new_top * 255).astype(np.uint8))
                    im.save("./Prerecorded_Video_Heatmap_static/Top_View"+str(ind)+".png")
                    im = cv2.imread(("./Prerecorded_Video_Heatmap_static/Top_View"+str(ind)+".png"))
                    video.write(im)
                    # ind = ind+1
                    ind = ind + 1
                    # break`


                    # top_view_sol = cv2.cvtColor(top_view_sol, cv2.COLOR_RGBA2RGB)

                    # rgbImage = cv2.cvtColor(npimage, cv2.COLOR_RGBA2RGB)


                    # cv2.circle(top_view_sol, (int(circle_coord[0][0][0]),int(circle_coord[0][0][1])), 10, COLORS[0], thickness=10, lineType=8, shift=0)


                    # plt.imshow(top_view_sol)
                    # print(img.shape[:2][::-1])
                    # cv2.imshow("top View",top_view_sol)
                    # cv2.waitKey()

                    # images_array.append(top_view_sol)
                    # optputFile3.write(top_view_sol)

                    # cv2.imwrite("Prerecorded_Video_Frames/TopView_PreRecorded_Recording_"+str(ind)+".png",top_view_sol)
                    # ind = ind +1

                    # display (or save) images
                    cv2.imshow(windowName4, new_top)
                    cv2.imshow(windowName, img_detect)
                    cv2.imshow(windowName2, img2_detect)
                    cv2.imshow(windowName3, img3_detect)
                    # cv2.waitKey()
                    # cv2.imshow('ipm', top_view_sol)
                    if cv2.waitKey(1) == 27:
                        break

                # for imagee in images_array:
                #     optputFile3.write(imagee)
                
                

                imgs_list = glob.glob("./Prerecorded_Video_Heatmap_static/*.png")
                for imgs in imgs_list:
                    img = cv2.imread(imgs)
                    video.write(img)

                  

                cap1.release()
                # optputFile1.release()

                cap2.release()
                # optputFile2.release()

                cap3.release()
            # optputFile3.release()
                cv2.destroyAllWindows()

            elif option_heat == "a":



                cap1 = cv2.VideoCapture('Recordings/Stream1Recording.avi')
                cap2 = cv2.VideoCapture('Recordings/Stream2Recording.avi')
                cap3 = cv2.VideoCapture('Recordings/Stream3Recording.avi')
                top_view = cv2.imread('Top_View.jpeg')
                size = (int(top_view.shape[1]),int(top_view.shape[0]))
                # optputFile3 = cv2.VideoWriter(
                    # 'TopView_PreRecorded_Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

                windowName = "Sample Feed from Camera 1"
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

                windowName2 = "Sample Feed from Camera 2"
                cv2.namedWindow(windowName2,cv2.WINDOW_NORMAL)

                windowName3 = "Sample Feed from Camera 3"
                cv2.namedWindow(windowName3,cv2.WINDOW_NORMAL)

                windowName4 = "Top View Of all Cameras"
                cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter('./Recordings/Animated_Heatmap_Prerecorded.avi', fourcc, 1, size)

                if (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                    ret = True
                else:
                    ret = False

                ind = 0
                images_array = []
                heatmap_array = []
                frames_to_consider = 5

                ind_bb = 0


                while (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                    ret,img = cap1.read()
                    ret1,img2 = cap2.read()
                    ret2,img3 = cap3.read()
                    top_view = cv2.imread('Top_View.jpeg')
                    # img = cv2.imread("00000160.jpg")
                    # img2 = cv2.imread("00000151.jpg")
                    # img3 = cv2.imread("Stream1-2.jpg")
                    # print(img3)

                    th,tw,_ = top_view.shape
                    # print(th,tw)

                    h,w,n = img3.shape
                    # img3 = img3[377:h][0:w]
                    # cv2.imwrite("cap_1.jpg",img)
                    # cv2.imwrite("cap_2.jpg",img2)
                    # cv2.imwrite("cap_3.jpg",img3)
                    # break


                # img = cv2.imread('img_h.jpeg')
                # img2 = cv2.imread('img_t.jpeg')
                # # img = cv2.imread('00000287.jpg')
                # img3 = cv2.imread('img_0.jpeg')
                    # img = img2

                    boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)


                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                
                    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                    path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                    f = open(path,'a')
                    # print(indices)
                    # print(boxes)
                    # print(ipm_matrix3)
                    circle_coord = []

                    toDraw = []
                    count = [0, 0, 0]
                    # break
                    img_detect = img.copy()
                    for i in indices:
                        # i = i[0]
                        box = boxes[i]
                        # print(ipm_matrix3*box)

                        # break
                        x = box[0]
                        # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = box[2]
                        h = box[3]
                        
                        # print(x,y)
                        #draw_bounding_box_base(img, class_ids[i], confidences[i], round(x), round(y+200), round(x+w), round(y+h),classes,COLORS)
                        toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                        count[0] = count[0] + 1
                        draw_bounding_box(img_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                        f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")

                        # cv2.circle(img, (round(x),round(y)), 10, COLORS[0], thickness=1, lineType=8, shift=0)
                        # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # print("\nDisplaying Detection Results...")
                    # windowName_detect = "Detection Result"
                    # cv2.namedWindow(windowName_detect)
                    # cv2.imshow(windowName_detect, img_detect)




                    boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img2, cfg, weights,classes)


                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                
                    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                    path = "./Bounding_Boxes_Files/BB_prerecorded_S2"+str(ind_bb)+".txt"
                    f = open(path,'a')
                    # print(indices)
                    # print(boxes)
                    # print(ipm_matrix3)
                    circle_coord = []

                    # break
                    img2_detect = img2.copy()
                    for i in indices:
                        # i = i[0]
                        box = boxes[i]
                        # print(ipm_matrix3*box)

                        # break
                        x = box[0]
                        # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = box[2]
                        h = box[3]
                        
                        # print(x,y)
                        #draw_bounding_box_base(img2, class_ids[i], confidences[i], round(x), round(y+170), round(x+w), round(y+h),classes,COLORS)
                        toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                        count[1] = count[1] + 1
                        draw_bounding_box(img2_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                        f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")







                    
                    boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img3, cfg, weights,classes)


                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                
                    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                    path = "./Bounding_Boxes_Files/BB_prerecorded_S3"+str(ind_bb)+".txt"
                    f = open(path,'a')
                    # print(indices)
                    # print(boxes)
                    # print(ipm_matrix3)
                    circle_coord = []

                    # break
                    img3_detect = img3.copy()
                    for i in indices:
                        # i = i[0]
                        box = boxes[i]
                        # print(ipm_matrix3*box)

                        # break
                        x = box[0]
                        # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = box[2]
                        h = box[3]
                        
                        # print(x,y)
                        #draw_bounding_box_base(img3, class_ids[i], confidences[i], round(x), round(y+120), round(x+w), round(y+h),classes,COLORS)
                        toDraw.append([round(x), round(y+120), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                        count[2] = count[2] + 1
                        draw_bounding_box(img3_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                        f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")


                    ipm_mat = []
                    ind_bb = ind_bb +1

                    # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                    # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                    # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                    # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                    # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                    pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                    pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                    pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                    pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                    # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                    pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                    # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                    # colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]
                    # i = 0
                    # for pt in pts3:
                    #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                    #     i = i+1
                    #     if i == len(colors):
                    #         i = 0

                    ## compute IPM matrix and apply it
                    ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                    # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                    # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                    ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                    # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                    ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                    # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                    ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                    ipm_mat.append(ipm1)
                    # cv2.imwrite('img1_transformed.jpg',ipm1)



                    # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                    # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                    ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                    ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                    # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                    ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                    ipm_mat.append(ipm2)
                    # cv2.imwrite('img2_transformed.jpg',ipm2)
                    # ipm_pts3 = np.array([[501,452],[655,452], [662,350], [507,350]], dtype=np.float32)
                    # ipm_matrix = cv2.getPerspectiveTransform(pts3, ipm_pts3)
                    # ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix, top_view.shape[:2][::-1])
                    # ipm_mat.append(ipm3)


                    # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                    ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                    # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                    ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                    # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                    ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                    ipm_mat.append(ipm3)
                    # cv2.imwrite('img3_transformed.jpg',ipm3)


                    toDrawPts = []
                    for i in range(0, count[0]):
                        #xpt = (toDraw[i][0] + toDraw[i][2])/2
                        #ypt = (toDraw[i][1] + toDraw[i][3])/2
                        #inter = np.matmul(ipm_matrix, np.array([round(xpt), round(ypt), 1]))
                        inter = np.matmul(ipm_matrix, np.array([toDraw[i][0], toDraw[i][1], 1]))
                        inter[0] = inter[0]/inter[2]
                        inter[1] = inter[1]/inter[2]
                        inter[2] = 1
                        toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                    for i in range(count[0], count[0]+count[1]):
                        #xpt = (toDraw[i][0] + toDraw[i][2])/2
                        #ypt = (toDraw[i][1] + toDraw[i][3])/2
                        #inter = np.matmul(ipm_matrix2, np.array([round(xpt), round(ypt), 1]))
                        inter = np.matmul(ipm_matrix2, np.array([toDraw[i][0], toDraw[i][1], 1]))
                        inter[0] = inter[0]/inter[2]
                        inter[1] = inter[1]/inter[2]
                        inter[2] = 1
                        toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                    for i in range(count[0]+count[1], count[0]+count[1]+count[2]):
                        #xpt = (toDraw[i][0] + toDraw[i][2])/2
                        #ypt = (toDraw[i][1] + toDraw[i][3])/2
                        #inter = np.matmul(ipm_matrix3, np.array([round(xpt), round(ypt), 1]))
                        inter = np.matmul(ipm_matrix3, np.array([toDraw[i][0], toDraw[i][1], 1]))
                        inter[0] = inter[0]/inter[2]
                        inter[1] = inter[1]/inter[2]
                        inter[2] = 1
                        toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])


                    #print("SHAPE", toDrawPts[0].shape, toDrawPts[0][0], toDrawPts[0][1], toDrawPts[0][2])
                    # ipm_mat.append(ipm3)




                    # print(ipm_mat)
                    images_array = [img,img2,img3]

                    # overlay = Image.open('img3_transformed.jpg').convert('RGBA')
                    # image = Image.open('Top_View.jpeg').convert('RGBA')
                    # image.paste(overlay, mask=overlay)
                    # top_view_sol = image
                    # image.save('result.png')




                    top_view = cv2.cvtColor(cv2.imread('Top_View.jpeg'),cv2.COLOR_BGR2RGB)/255
                    h,w,n = top_view.shape
                    top_view_sol = top_view.copy()
                    top_view_sol = toRGBA(top_view_sol)
                    # cv2.imshow("Top View Actual", top_view)
                    # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                    # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                    # (status, stitched) = stitcher.stitch(images_array)

                    # cv2.imshow("stitched version", stitched)
                    # ipm_mat = ipm_mat[0]

                    # print(circle_coord[0][0])

                    for ipm in ipm_mat:
                        # ipm = toRGBA(ipm)
                        # print("Here")
                        for i in range(h):
                          for j in range(w):
                            location_px = ipm[i][j].copy()
                            if location_px[3] == 1:
                                 # and (top_view_sol[i][j]>0.7).all())
                              #  and (location_px <0.1).any():
                              top_view_sol[i][j] = location_px


                    for i in range(len(toDrawPts)):
                        draw_bounding_box_base_pt(top_view_sol, toDrawPts[i][2], toDrawPts[i][0], toDrawPts[i][1], toDrawPts[i][3], toDrawPts[i][4])

                                   # cv2.cvtColor(top_view_sol,cv2.COLOR_BGR2RGB)/255


                    


                    k = 21
                        # x=0
                    gauss = cv2.getGaussianKernel(k,np.sqrt(64))
                    gauss = gauss * gauss.T
                    gauss = (gauss/gauss[int(k/2),int(k/2)])

                    heatmap = np.zeros((top_view.shape[0],top_view.shape[1],3)).astype(np.float32)
                    # heatmap = toRGBA(heatmap)
                    # heatmap[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])] += 1
                    # heatmap[heatmap > 0.8] = 0


                    j = cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),cv2.COLORMAP_SUMMER),cv2.COLOR_BGR2RGB).astype(np.float32)/255
                    points = toDrawPts

                    for p in points:
                        # print(p[0]- int(k/2))
                        # print(p[0]+int(k/2)+1)
                        # print(p[1]- int(k/2))
                        # print(p[1]+int(k/2)+1)

                        # print(heatmap[int(p[0]):int(p[0]+int(k/2)+1),int(p[1]-int(k/2)):int(p[1]+int(k/2)+1),:].shape)
                        # print(j.shape)



                        try:
                            # b = heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:]
                            # c = j + b

                            # heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:] = j
                            draw_bounding_box_base_pt(heatmap, p[2], p[0], p[1], p[3], p[4])
                        except:
                            pass

                    # m = np.max(heatmap, axis = (0,1))
                    # heatmap = heatmap/m




                    heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)

                    mask = np.where(heatmap > 0.2,1,0).astype(np.float32)
                    inv_mask = np.ones((top_view.shape[0],top_view.shape[1],3))*(1 - mask)[:,:,None]
                    
                    im = Image.fromarray((heatmap * 255).astype(np.uint8))
                    im.save('./frames_S1/frames'+str(ind)+".png")


                    images = cv2.imread('./frames_S1/frames'+str(ind)+".png", 0)
                    # colormap = plt.get_cmap('inferno')
                    # heatmap = (colormap(images) * 2**16).astype(np.uint16)[:,:,:3]
                    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

                    # heatmap = (heatmap/255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(images, cv2.COLORMAP_AUTUMN)
                    heatmap_array.append(heatmap)
                    # # imask = (inv_mask/255).astype(np.uint8)
                    # # imask = cv2.applyColorMap(imask, cv2.COLORMAP_HOT)
                    top_view = rgba2rgb(top_view)

                    
                    new_top = np.where(inv_mask>0,top_view,inv_mask)
                    new_top = inv_mask * top_view

                    try:
                        heatmap_new = heatmap_array[len(heatmap_array) - frames_to_consider]
                        for i in range(len(heatmap_array) - frames_to_consider+1,len(heatmap_array)):
                            heatmap_new = heatmap_new+heatmap_array[i]
                    except:
                        heatmap_new = heatmap_array[0]
                        for i in range(1,len(heatmap_array)):
                            heatmap_new = heatmap_new+heatmap_array[i]


                    new_top = new_top + heatmap_new





                    # cv2.imshow("New_Top_View",new_top)
                    # cv2.imshow("Heatmap",heatmap)
                    # cv2.imshow("Top View Sol",top_view_sol)
                    # cv2.imshow("Image",img_detect)



                    # cv2.imshow("Actual img",img)

                    # cv2.waitKey()
                    # ind = ind + 1

                    

                    im = Image.fromarray((new_top * 255).astype(np.uint8))
                    im.save("./Prerecorded_Video_Heatmap_Animated/Top_View"+str(ind)+".png")
                    im = cv2.imread(("./Prerecorded_Video_Heatmap_Animated/Top_View"+str(ind)+".png"))
                    video.write(im)
                    ind = ind+1
                    # break`


                    # top_view_sol = cv2.cvtColor(top_view_sol, cv2.COLOR_RGBA2RGB)

                    # rgbImage = cv2.cvtColor(npimage, cv2.COLOR_RGBA2RGB)


                    # cv2.circle(top_view_sol, (int(circle_coord[0][0][0]),int(circle_coord[0][0][1])), 10, COLORS[0], thickness=10, lineType=8, shift=0)


                    # plt.imshow(top_view_sol)
                    # print(img.shape[:2][::-1])
                    # cv2.imshow("top View",top_view_sol)
                    # cv2.waitKey()

                    # images_array.append(top_view_sol)
                    # optputFile3.write(top_view_sol)

                    # cv2.imwrite("Prerecorded_Video_Frames/TopView_PreRecorded_Recording_"+str(ind)+".png",top_view_sol)
                    # ind = ind +1

                    # display (or save) images
                    cv2.imshow(windowName4, new_top)
                    cv2.imshow(windowName, img_detect)
                    cv2.imshow(windowName2, img2_detect)
                    cv2.imshow(windowName3, img3_detect)
                    # cv2.waitKey()
                    # cv2.imshow('ipm', top_view_sol)
                    if cv2.waitKey(1) == 27:
                        break

                # for imagee in images_array:
                #     optputFile3.write(imagee)
                
                

                imgs_list = glob.glob("./Prerecorded_Video_Frames/*.png")
                for imgs in imgs_list:
                    img = cv2.imread(imgs)
                    video.write(img)

                  

                cap1.release()
                # optputFile1.release()

                cap2.release()
                # optputFile2.release()

                cap3.release()
            # optputFile3.release()
                cv2.destroyAllWindows()

            elif option_heat == "c":
                cap1 = cv2.VideoCapture('Recordings/Stream1Recording.avi')
                cap2 = cv2.VideoCapture('Recordings/Stream2Recording.avi')
                cap3 = cv2.VideoCapture('Recordings/Stream3Recording.avi')
                top_view = cv2.imread('Top_View.jpeg')
                size = (int(top_view.shape[1]),int(top_view.shape[0]))
                # optputFile3 = cv2.VideoWriter(
                    # 'TopView_PreRecorded_Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

                windowName = "Sample Feed from Camera 1"
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

                windowName2 = "Sample Feed from Camera 2"
                cv2.namedWindow(windowName2,cv2.WINDOW_NORMAL)

                windowName3 = "Sample Feed from Camera 3"
                cv2.namedWindow(windowName3,cv2.WINDOW_NORMAL)

                windowName4 = "Top View Of all Cameras"
                cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter('./Recordings/COVID_Heatmap_Prerecorded.avi', fourcc, 1, size)

                if (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                    ret = True
                else:
                    ret = False

                ind = 0
                images_array = []
                heatmap_array = []
                ind_bb = 0
                frames_to_consider = 5

                px_img1 = (((1099-863) *2)/1.245)

                px_img2 = (((357-198) *2)/1.245)

                px_img3 = (((1297-1162) *2)/1.245)


                while (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                    ret,img = cap1.read()
                    ret1,img2 = cap2.read()
                    ret2,img3 = cap3.read()
                    top_view = cv2.imread('Top_View.jpeg')
                    # img = cv2.imread("00000160.jpg")
                    # img2 = cv2.imread("00000151.jpg")
                    # img3 = cv2.imread("Stream1-2.jpg")
                    # print(img3)

                    th,tw,_ = top_view.shape
                    # print(th,tw)

                    h,w,n = img3.shape
                    # img3 = img3[377:h][0:w]
                    # cv2.imwrite("cap_1.jpg",img)
                    # cv2.imwrite("cap_2.jpg",img2)
                    # cv2.imwrite("cap_3.jpg",img3)
                    # break


                # img = cv2.imread('img_h.jpeg')
                # img2 = cv2.imread('img_t.jpeg')
                # # img = cv2.imread('00000287.jpg')
                # img3 = cv2.imread('img_0.jpeg')
                    # img = img2



                    # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)

                    width = img.shape[1]
                    height = img.shape[0]

                    scale = 0.00392

                    COLORS = np.random.uniform(0, 255, size=(len(classes_person), 3))
                    # COLORS = colors
                    

                    net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                    blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
                    # set input blob for the network
                    net.setInput(blob)

                    outs = net.forward(output_layers)
                    # print(outs)
                    


                    class_ids = []
                    confidences = []
                    boxes = []
                    conf_threshold = 0.5
                    nms_threshold = 0.7

                    

                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            
                            class_idx = np.argmax(scores)
                            # print(classval)
                            confidence = scores[class_idx]
                            # print(confidence)
                            if confidence > 0.5:

                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                x = center_x - w / 2
                                y = center_y - h / 2
                                # class_idx = np.argmax(scores)
                                if class_idx > (len(classes) - 1):
                                    class_idx = class_idx % (len(classes)+1)
                                # print(classval)
                                if class_idx == 0:
                                    class_ids.append(class_idx)
                                    confidences.append(float(confidence))
                                    boxes.append([x, y, w, h])

                    # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                    classes_covid = ["violation","no_violation"]
                    # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                    path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                    f = open(path,'a')
                    # print(indices)
                    # print(boxes)
                    # print(ipm_matrix3)
                    circle_coord = []
                    toDraw = []
                    # color_array = []
                    count = [0, 0, 0]

                    # break
                    img_detect = img.copy()
                    box = [boxes[i] for i in indices]
                    # print(box)
                    class_id_covid = []
                    # if len(box) == 1:


                    for i in range(len(box)-1):
                        if i + 1 != len(box):
                            dist = box[i][0] - box[i+1][0]
                            if dist > px_img1:
                                class_id_covid.append(1)
                            else:
                                class_id_covid.append(0)
                            x = box[i][0]
                            # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            y =  box[i][1]
                            # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            w = (box[i][2] + box[i+1][2])
                            h = (box[i][3])
                        else:
                            class_id_covid.append(1)

                        

                        # x = box[i][0]
                        # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        # y =  box[i][1]
                        # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        # w = box[i][2]
                        # h = box[i][3]
                        # draw_bounding_box_base(img, class_id_covid[i], confidences[i], round(x+100), round(y+200), round(x+w), round(y+h),classes_covid,COLORS)
                        toDraw.append([round(x+150), round(y+200), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                        # color_array.append(COLORS[class_ids[i]])
                        count[0] = count[0] + 1
                        draw_bounding_box(img_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+(w)), round(y+(h)),classes_covid,COLORS)



                    net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                    blob = cv2.dnn.blobFromImage(img2, scale, (416,416), (0,0,0), True, crop=False)
                    # set input blob for the network
                    net.setInput(blob)

                    outs = net.forward(output_layers)
                    # print(outs)
                    


                    class_ids = []
                    confidences = []
                    boxes = []
                    conf_threshold = 0.95
                    nms_threshold = 0.7

                    

                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            
                            class_idx = np.argmax(scores)
                            # print(class_idx)
                            # print(classval)
                            confidence = scores[class_idx]
                            # print(confidence)
                            if confidence > 0.5:

                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                x = center_x - w / 2
                                y = center_y - h / 2
                                # class_idx = np.argmax(scores)
                                if class_idx > (len(classes) - 1):
                                    class_idx = class_idx % (len(classes)+1)
                                # print(classval)
                                if class_idx == 0:
                                    class_ids.append(class_idx)
                                    confidences.append(float(confidence))
                                    boxes.append([x, y, w, h])


                    # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img2, cfg, weights,classes)
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                    classes_covid = ["violation","no_violation"]
                    # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                    path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                    f = open(path,'a')
                    # print(indices)
                    # print(boxes)
                    # print(ipm_matrix3)
                    circle_coord = []

                    # break
                    img2_detect = img2.copy()
                    box = [boxes[i] for i in indices]
                    # print(box)
                    class_id_covid = []
                    # if len(box) == 1:


                    for i in range(len(box)-1):
                        if i + 1 != len(box):
                            dist = box[i][0] - box[i+1][0]
                            if dist > px_img2:
                                class_id_covid.append(1)
                            else:
                                class_id_covid.append(0)
                            x = box[i][0]
                            # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            y =  box[i][1]
                            # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            w = box[i][2] + box[i+1][2]
                            h = box[i][3] 
                            # h = h + 50
                            # y = y + 50
                            # x = x+ 50
                        else:
                            class_id_covid.append(1)

                        

                        # x = box[i][0]
                        # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        # y =  box[i][1]
                        # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        # w = box[i][2]
                        # h = box[i][3]
                        # draw_bounding_box_base(img2, class_id_covid[i], confidences[i], round(x+130), round(y+200), round(x+w), round(y+h),classes_covid,COLORS)
                        toDraw.append([round(x+130), round(y+200), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                        # color_array.append(COLORS[class_ids[i]])
                        count[1] = count[1] + 1
                        draw_bounding_box(img2_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes_covid,COLORS)


                    # net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                    # layer_names = net.getLayerNames()
                    # output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                    blob = cv2.dnn.blobFromImage(img3, scale, (416,416), (0,0,0), True, crop=False)
                    # set input blob for the network
                    net.setInput(blob)

                    outs = net.forward(output_layers)
                    # print(outs)
                    


                    class_ids = []
                    confidences = []
                    boxes = []
                    conf_threshold = 0.5
                    nms_threshold = 0.7

                    

                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            
                            class_idx = np.argmax(scores)
                            # print(classval)
                            confidence = scores[class_idx]
                            # print(confidence)
                            if confidence > 0.5:

                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                x = center_x - w / 2
                                y = center_y - h / 2
                                # class_idx = np.argmax(scores)
                                if class_idx > (len(classes) - 1):
                                    class_idx = class_idx % (len(classes)+1)
                                # print(classval)
                                if class_idx == 0:
                                    class_ids.append(class_idx)
                                    confidences.append(float(confidence))
                                    boxes.append([x, y, w, h])


                    # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img3, cfg, weights,classes)
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)


                    classes_covid = ["violation","no_violation"]
                    # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                    path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                    f = open(path,'a')
                    # print(indices)
                    # print(boxes)
                    # print(ipm_matrix3)
                    circle_coord = []

                    # break
                    img3_detect = img3.copy()
                    box = [boxes[i] for i in indices]
                    # print(box)
                    # print("confidence::", confidences)
                    class_id_covid = []
                    # if len(box) == 1:


                    for i in range(len(box)-1):
                        if i + 1 != len(box):
                            dist = box[i][0] - box[i+1][0]
                            if dist > px_img2:
                                class_id_covid.append(1)
                            else:
                                class_id_covid.append(0)
                            x = box[i][0]
                            # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            y =  box[i][1]
                            # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            w = box[i][2] + box[i+1][2]
                            h = box[i][3]
                        else:
                            class_id_covid.append(1)

                        

                        # x = box[i][0]
                        # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        # y =  box[i][1]
                        # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        # w = box[i][2]
                        # h = box[i][3]
                        # draw_bounding_box_base(img3, class_id_covid[i], confidences[i], round(x+50), round(y+180), round(x+w), round(y+h),classes_covid,COLORS)
                        toDraw.append([ round(x+50), round(y+180), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                        # color_array.append(COLORS[class_ids[i]])
                        count[2] = count[2] + 1
                        draw_bounding_box(img3_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes_covid,COLORS)
                    


                    ipm_mat = []
                    ind_bb = ind_bb +1

                    # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                    # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                    # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                    # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                    # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                    pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                    pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                    pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                    pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                    # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                    pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                    # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                    colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]
                    # i = 0
                    # for pt in pts3:
                    #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                    #     i = i+1
                    #     if i == len(colors):
                    #         i = 0

                    ## compute IPM matrix and apply it
                    ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                    # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                    # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                    ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                    # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                    ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                    # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                    ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                    ipm_mat.append(ipm1)
                    # cv2.imwrite('img1_transformed.jpg',ipm1)



                    # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                    # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                    ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                    ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                    # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                    ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                    ipm_mat.append(ipm2)
                  


                    # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                    ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                    # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                    ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                    # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                    ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                    ipm_mat.append(ipm3)
                    # cv2.imwrite('img3_transformed.jpg',ipm3)


                    
                    # ipm_mat.append(ipm3)


                    toDrawPts = []
                    for i in range(0, count[0]):
                        #xpt = (toDraw[i][0] + toDraw[i][2])/2
                        #ypt = (toDraw[i][1] + toDraw[i][3])/2
                        #inter = np.matmul(ipm_matrix, np.array([round(xpt), round(ypt), 1]))
                        inter = np.matmul(ipm_matrix, np.array([toDraw[i][0], toDraw[i][1], 1]))
                        inter[0] = inter[0]/inter[2]
                        inter[1] = inter[1]/inter[2]
                        inter[2] = 1
                        toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                    for i in range(count[0], count[0]+count[1]):
                        #xpt = (toDraw[i][0] + toDraw[i][2])/2
                        #ypt = (toDraw[i][1] + toDraw[i][3])/2
                        #inter = np.matmul(ipm_matrix2, np.array([round(xpt), round(ypt), 1]))
                        inter = np.matmul(ipm_matrix2, np.array([toDraw[i][0], toDraw[i][1], 1]))
                        inter[0] = inter[0]/inter[2]
                        inter[1] = inter[1]/inter[2]
                        inter[2] = 1
                        toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                    for i in range(count[0]+count[1], count[0]+count[1]+count[2]):
                        #xpt = (toDraw[i][0] + toDraw[i][2])/2
                        #ypt = (toDraw[i][1] + toDraw[i][3])/2
                        #inter = np.matmul(ipm_matrix3, np.array([round(xpt), round(ypt), 1]))
                        inter = np.matmul(ipm_matrix3, np.array([toDraw[i][0], toDraw[i][1], 1]))
                        inter[0] = inter[0]/inter[2]
                        inter[1] = inter[1]/inter[2]
                        inter[2] = 1
                        toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])


                    #print("SHAPE", toDrawPts[0].shape, toDrawPts[0][0], toDrawPts[0][1], toDrawPts[0][2])




                    # print(ipm_mat)
                    images_array = [img,img2,img3]

               




                    h,w,n = top_view.shape
                    top_view_sol = top_view.copy()
                    top_view_sol = toRGBA(top_view_sol)
                    top_view_sol_copy = top_view_sol.copy()
                    # cv2.imshow("Top View Actual", top_view)
                    # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                    # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                    # (status, stitched) = stitcher.stitch(images_array)

                    # cv2.imshow("stitched version", stitched)
                    # ipm_mat = ipm_mat[0]

                    # print(circle_coord[0][0])

                    for ipm in ipm_mat:
                        # ipm = toRGBA(ipm)
                        # print("Here")
                        for i in range(h):
                          for j in range(w):
                            location_px = ipm[i][j].copy()
                            if location_px[3] == 1:
                                 # and (top_view_sol[i][j]>0.7).all())
                              #  and (location_px <0.1).any():
                              top_view_sol[i][j] = location_px



                    for i in range(len(toDrawPts)):
                        draw_bounding_box_base_pt(top_view_sol, toDrawPts[i][2], toDrawPts[i][0], toDrawPts[i][1], toDrawPts[i][3], toDrawPts[i][4])

                    top_view = cv2.cvtColor(cv2.imread('Top_View.jpeg'),cv2.COLOR_BGR2RGB)/255
                                   # cv2.cvtColor(top_view_sol,cv2.COLOR_BGR2RGB)/255


                    # j = cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),cv2.COLORMAP_SUMMER),cv2.COLOR_BGR2RGB).astype(np.float32)/255
                    k = 21
                        # x=0
                    gauss = cv2.getGaussianKernel(k,np.sqrt(64))
                    gauss = gauss * gauss.T
                    gauss = (gauss/gauss[int(k/2),int(k/2)])

                    heatmap = np.zeros((top_view.shape[0],top_view.shape[1],3)).astype(np.float32)
                    points = toDrawPts
                    # print(points)

                    for p in points:
                        # print(p[0]- int(k/2))
                        # print(p[0]+int(k/2)+1)
                        # print(p[1]- int(k/2))
                        # print(p[1]+int(k/2)+1)

                        # print(heatmap[int(p[0]):int(p[0]+int(k/2)+1),int(p[1]-int(k/2)):int(p[1]+int(k/2)+1),:].shape)
                        # print(j.shape)



                        try:
                            # b = heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:]
                            # c = j + b

                            # heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:] = j
                            draw_bounding_box_base_pt(heatmap, p[2], p[0], p[1], p[3], p[4])
                        except:
                            pass

                    # m = np.max(heatmap, axis = (0,1))
                    # heatmap = heatmap/m




                    heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)

                    mask = np.where(heatmap > 0.2,1,0).astype(np.float32)
                    inv_mask = np.ones((top_view.shape[0],top_view.shape[1],3))*(1 - mask)[:,:,None]
                    
                    im = Image.fromarray((heatmap * 255).astype(np.uint8))
                    im.save('./frames_S1/frames'+str(ind)+".png")


                    images = cv2.imread('./frames_S1/frames'+str(ind)+".png", 0)
                    # colormap = plt.get_cmap('inferno')
                    # heatmap = (colormap(images) * 2**16).astype(np.uint16)[:,:,:3]
                    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

                    # heatmap = (heatmap/255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(images, cv2.COLORMAP_AUTUMN)
                    heatmap_array.append(heatmap)
                    # # imask = (inv_mask/255).astype(np.uint8)
                    # # imask = cv2.applyColorMap(imask, cv2.COLORMAP_HOT)
                    top_view = rgba2rgb(top_view)

                    
                    new_top = np.where(inv_mask>0,top_view,inv_mask)
                    new_top = inv_mask * top_view

                    try:
                        heatmap_new = heatmap_array[len(heatmap_array) - frames_to_consider]
                        for i in range(len(heatmap_array) - frames_to_consider+1,len(heatmap_array)):
                            heatmap_new = heatmap_new+heatmap_array[i]
                    except:
                        heatmap_new = heatmap_array[0]
                        for i in range(1,len(heatmap_array)):
                            heatmap_new = heatmap_new+heatmap_array[i]


                    new_top = new_top + heatmap_new



                    im = Image.fromarray((new_top * 255).astype(np.uint8))
                    im.save("./Prerecorded_Video_Heatmap_Covid/Top_View"+str(ind)+".png")
                    im = cv2.imread(("./Prerecorded_Video_Heatmap_Covid/Top_View"+str(ind)+".png"))
                    video.write(im)
                    ind = ind+1


                    cv2.imshow(windowName4,new_top)
                    cv2.imshow(windowName2,img3_detect)
                    cv2.imshow(windowName3,img2_detect)
                    cv2.imshow(windowName,img_detect)

                    if cv2.waitKey(1) == 27:
                        break



                    # cv2.imshow("Actual img",img)
                    # im = Image.fromarray((heatmap * 255).astype(np.uint8))
                    # im.save('./frames_S1/frames'+str(ind)+".png")
                    # ind = ind + 1



        # Record video
        else:
            windowName = "Sample Feed from Camera 1"
            cv2.namedWindow(windowName)

            windowName2 = "Sample Feed from Camera 2"
            cv2.namedWindow(windowName2)

            windowName3 = "Sample Feed from Camera 3"
            cv2.namedWindow(windowName3)

            capture1 = cv2.VideoCapture(0)  # laptop's camera
            # capture2 = cv2.VideoCapture(0)  # laptop's camera
            # capture3 = cv2.VideoCapture(0)  # laptop's camera
            

            capture2 = cv2.VideoCapture("http://10.130.145.123:8080/video")
            capture3 = cv2.VideoCapture("http://10.130.145.123:8080/video")
            # capture2 = cv2.VideoCapture("http://10.130.12.104:8080/video")   # sample code for mobile camera video capture using IP camera
            # print("capturing")

            # define size for recorded video frame for video 1
            width1 = int(capture1.get(3))
            height1 = int(capture1.get(4))
            size1 = (width1, height1)

            width2 = int(capture2.get(3))
            height2 = int(capture2.get(4))
            size2 = (width2, height2)

            width3 = int(capture3.get(3))
            height3 = int(capture3.get(4))
            size3 = (width3, height3)

            # print("size Set")

            # frame of size is being created and stored in .avi file
            optputFile1 = cv2.VideoWriter(
                'Stream1Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)

            optputFile2 = cv2.VideoWriter(
                'Stream2Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size2)

            optputFile3 = cv2.VideoWriter(
                'Stream3Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size3)

            # check if feed exists or not for camera 1
            if (capture1.isOpened() and capture2.isOpened() and capture3.isOpened()):
                ret1, frame1 = capture1.read()
                ret2, frame2 = capture2.read()
                ret3, frame3 = capture3.read()
                # print("Checked!")
            else:
                ret1 = False
                ret2 = False
                ret3 = False

            start = time.time()
            # print("start",start)


            while (ret1 and ret2 and ret3):
                ret1, frame1 = capture1.read()
                ret2, frame2 = capture2.read()
                ret3, frame3 = capture3.read()
                # sample feed display from camera 1




                cv2.imshow(windowName, frame1)
                # cv2.waitKey(0)
                cv2.imshow(windowName2, frame2)
                # cv2.waitKey(0)
                cv2.imshow(windowName3, frame3)
                # cv2.waitKey(0)
                
                # print(time.time())

                # saves the frame from camera 1
                optputFile1.write(frame1)
                optputFile2.write(frame2)
                optputFile3.write(frame3)

                # escape key (27) to exit
                if cv2.waitKey(1) == 27:
                    break
                #elif ((time.time() - start) == 60) or ((time.time() - start) > 60):
                #    break

            capture1.release()
            optputFile1.release()

            capture2.release()
            optputFile2.release()

            capture3.release()
            optputFile3.release()
            cv2.destroyAllWindows()

    elif option == 2:

        weights = "mask-yolov3_20000.weights"
        weights_2 = weights
        weights_3 = weights
        
        print("Press Y for Mask/No mask Detection\nPress P for Top View Projection\nPress d for Top View Projection and Detection\nPress s for Covid Violation detections \nPress h for heatmaps ")
        option_yolo = str(input())
        # live stream
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        if option_yolo != "y":

            windowName1 = "Live Stream Camera 1"
            cv2.namedWindow(windowName1,cv2.WINDOW_NORMAL)

            windowName2 = "Live Stream Camera 2"
            cv2.namedWindow(windowName2,cv2.WINDOW_NORMAL)

            windowName3 = "Live Stream Camera 3"
            cv2.namedWindow(windowName3,cv2.WINDOW_NORMAL)

        capture1 = cv2.VideoCapture(0)  # laptop's camera
        # capture2 = cv2.VideoCapture(0)  # laptop's camera
        # capture3 = cv2.VideoCapture(0)  # laptop's camera
        

        capture2 = cv2.VideoCapture("http://10.130.145.123:8080/video")
        capture3 = cv2.VideoCapture("http://10.130.145.123:8080/video")
        # capture2 = cv2.VideoCapture("http://10.130.145.123:8080/video")
        # capture3 = cv2.VideoCapture("http://10.130.12.131:8080/video")


        if (capture1.isOpened() and capture2.isOpened() and capture3.isOpened()):
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
        else:
            ret1 = False
            ret2 = False
            ret3 = False


        while (ret1 and ret2 and ret3):


            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
            # sample feed display from camera 1

            

            if option_yolo =="y":

                img1 = frame1
                # img1 = cv2.imread('C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/Yolo detector/Yolo-Annotation-Tool-New--master_Stream1/Images/stream1/00000152.jpg')
                Width = int(capture1.get(3))
                Height = int(capture1.get(4))
                scale = 0.00392

                
                with open(classes_file, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]

                print(classes)

                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

                net = cv2.dnn.readNetFromDarknet(cfg, weights)

                blob = cv2.dnn.blobFromImage(img1, scale, (Width,Height), (0,0,0), True, crop=False)
                # set input blob for the network
                net.setInput(blob)
                outs = net.forward(get_output_layers(net))
                # print(outs)
                


                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.1
                nms_threshold = 0.1

                # for each detetion from each output layer 
                # get the confidence, class id, bounding box params
                # and ignore weak detections (confidence < 0.5)
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        
                        class_idx = np.argmax(scores)
                        # print(classval)
                        confidence = scores[class_idx]
                        # print(confidence)
                        if confidence > 0.1:

                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            w = int(detection[2] * Width)
                            h = int(detection[3] * Height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            # class_idx = np.argmax(scores)
                            print(class_idx)
                            # if class_idx > (len(classes) - 1):
                            #     class_idx = class_idx % len(classes)
                            # print(classval)
                            class_ids.append(class_idx)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])


                

                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                # go through the detections remaining
                # after nms and draw bounding box
                path = os.path.join(os.getcwd()+"/Detections/BB_Live_S1.txt")
                f = open(path,'w')
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    
                    draw_bounding_box(img1, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                    f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")



                # display output image    
                windowName_detect = "Detection Result"
                cv2.namedWindow(windowName_detect)
                cv2.imshow(windowName_detect, img1)
                # cv2.waitKey(0)

                # break



                img2 = frame2
                # img1 = cv2.imread('C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/Yolo detector/Yolo-Annotation-Tool-New--master_Stream1/Images/stream1/00000152.jpg')
                Width2 = int(capture2.get(3))
                Height2 = int(capture2.get(4))
                scale = 0.00392

                
                with open(classes_file, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]

                print(classes)

                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

                net2 = cv2.dnn.readNetFromDarknet(cfg, weights_2)

                blob2 = cv2.dnn.blobFromImage(img2, scale, (416,416), (0,0,0), True, crop=False)
                # set input blob for the network
                net2.setInput(blob2)

                outs2 = net2.forward(get_output_layers(net2))
                # print(outs)
                


                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.5

                # for each detetion from each output layer 
                # get the confidence, class id, bounding box params
                # and ignore weak detections (confidence < 0.5)
                for out in outs2:
                    for detection in out:
                        scores = detection[5:]
                        
                        class_idx = np.argmax(scores)
                        # print(classval)
                        confidence = scores[class_idx]
                        # print(confidence)
                        if confidence > 0.5:

                            center_x = int(detection[0] * 416)
                            center_y = int(detection[1] * 416)
                            w = int(detection[2] * 416)
                            h = int(detection[3] * 416)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            # class_idx = np.argmax(scores)
                            print(class_idx)
                            # if class_idx > (len(classes) - 1):
                            #     class_idx = class_idx % len(classes)
                            # print(classval)
                            class_ids.append(class_idx)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])


                

                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                # go through the detections remaining
                # after nms and draw bounding box
                path = os.path.join(os.getcwd()+"/Detections/BB_Live_S2.txt")
                f = open(path,'w')
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    
                    draw_bounding_box(img2, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                    f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")

                # display output image    
                windowName_detect_2 = "Detection Result 2"
                cv2.namedWindow(windowName_detect_2)
                cv2.imshow(windowName_detect_2, img2)
                # cv2.waitKey(0)



                img3 = frame3
                # img3 = cv2.imread('C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/Yolo detector/Yolo-Annotation-Tool-New--master_Stream1/Images/stream1/00000152.jpg')
                Width3 = int(capture3.get(3))
                Height3 = int(capture3.get(4))
                scale = 0.00392

                
                with open(classes_file, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]

                print(classes)

                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

                net = cv2.dnn.readNetFromDarknet(cfg, weights_3)

                blob = cv2.dnn.blobFromImage(img3, scale, (416,416), (0,0,0), True, crop=False)
                # set input blob for the network
                net.setInput(blob)

                outs = net.forward(get_output_layers(net))
                # print(outs)
                


                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.1
                nms_threshold = 0.1

                # for each detetion from each output layer 
                # get the confidence, class id, bounding box params
                # and ignore weak detections (confidence < 0.5)
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        
                        class_idx = np.argmax(scores)
                        # print(classval)
                        confidence = scores[class_idx]
                        # print(confidence)
                        if confidence > 0.1:

                            center_x = int(detection[0] * 416)
                            center_y = int(detection[1] * 416)
                            w = int(detection[2] * 416)
                            h = int(detection[3] * 416)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            # class_idx = np.argmax(scores)
                            print(class_idx)
                            if class_idx > (len(classes) - 1):
                                class_idx = class_idx % len(classes)
                            # print(classval)
                            class_ids.append(class_idx)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])


                

                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                # go through the detections remaining
                # after nms and draw bounding box
                path = os.path.join(os.getcwd()+"/Detections/BB_Live_S3.txt")
                f3 = open(path,'w')
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    
                    draw_bounding_box(img3, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                    f3.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")

                # display output image    
                windowName_detect_3 = "Detection Result 3"
                cv2.namedWindow(windowName_detect_3)
                cv2.imshow(windowName_detect_3, img3)
                
                if cv2.waitKey(1) == 27:
                    break


            elif option_yolo == "p":
                windowName4 = "Top View from all cameras"
                cv2.namedWindow(windowName4,cv2.WINDOW_NORMAL)
                img = frame1
                img2 = frame2
                img3 = frame3
                h,w,n = img3.shape

                top_view = cv2.imread('Top_View.jpeg')

                # img3 = img3[377:h][0:w]
                # cv2.imwrite("cap_1.jpg",img)
                # cv2.imwrite("cap_2.jpg",img2)
                # cv2.imwrite("cap_3.jpg",img3)
                # break


            

                ipm_mat = []

                # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                # colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0)]
                # i = 0
                # for pt in pts:
                #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                #     i = i+1

                ## compute IPM matrix and apply it
                ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)
                ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                ipm_matrix,_ = cv2.findHomography(pts, ipm_pts)
                # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                ipm_mat.append(ipm1)



                # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)
                ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                ipm_mat.append(ipm2)
                # ipm_pts3 = np.array([[501,452],[655,452], [662,350], [507,350]], dtype=np.float32)
                # ipm_matrix = cv2.getPerspectiveTransform(pts3, ipm_pts3)
                # ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix, top_view.shape[:2][::-1])
                # ipm_mat.append(ipm3)


                # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)
                ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)
                ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                ipm_mat.append(ipm3)

                # print(ipm_mat)



                h,w,n = top_view.shape
                top_view_sol = top_view.copy()
                top_view_sol = toRGBA(top_view_sol)

                for ipm in ipm_mat:
                    # ipm = toRGBA(ipm)
                    # print("Here")
                    for i in range(h):
                      for j in range(w):
                        location_px = ipm[i][j].copy()
                        if location_px[3] == 1:
                          #  and (location_px <0.1).any():
                          top_view_sol[i][j] = location_px


                # plt.imshow(top_view_sol)
                # print(img.shape[:2][::-1])
                # cv2.imshow("top View",top_view_sol)
                # cv2.waitKey()

                # images_array.append(top_view_sol)
                # optputFile3.write(top_view_sol)

                # cv2.imwrite("Prerecorded_Video_Frames/TopView_PreRecorded_Recording_"+str(ind)+".png",top_view_sol)
                # ind = ind +1

                # display (or save) images
                cv2.imshow(windowName4, top_view_sol)
                cv2.imshow(windowName1, img)
                cv2.imshow(windowName2, img2)
                cv2.imshow(windowName3, img3)
                # cv2.waitKey()
                # cv2.imshow('ipm', top_view_sol)
                if cv2.waitKey(1) == 27:
                    break

            elif option_yolo == "d":


                windowName4 = "Top View from all cameras"
                cv2.namedWindow(windowName4,cv2.WINDOW_NORMAL)
                img = frame1
                img2 = frame2
                img3 = frame3
                h,w,n = img3.shape

                top_view = cv2.imread('Top_View.jpeg')



                boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)


                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            

            
                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./BB_Live_S1.txt"
                f = open(path,'w')
                # print("1",indices)
                # print("1",boxes)
                # print(ipm_matrix3)
                circle_coord = []

                # break
                img_detect = img.copy()
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    # print(ipm_matrix3*box)

                    # break
                    x = box[0]
                    # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    y =  box[1]
                    # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    w = box[2]
                    h = box[3]
                    
                    # print(x,y)
                    draw_bounding_box_base(img, class_ids[i], confidences[i], round(x), round(y+200), round(x+w), round(y+h),classes,COLORS)
                    draw_bounding_box(img_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                    f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")

                    # cv2.circle(img, (round(x),round(y)), 10, COLORS[0], thickness=1, lineType=8, shift=0)
                    # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # print("\nDisplaying Detection Results...")
                # windowName_detect = "Detection Result"
                # cv2.namedWindow(windowName_detect)
                # cv2.imshow(windowName_detect, img_detect)




                boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img2, cfg, weights,classes)


                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

       

            
                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./BB_Live_S2.txt"
                f = open(path,'w')
                # print("2",indices)
                # print("2",boxes)
                # print(ipm_matrix3)
                circle_coord = []

                # break
                img2_detect = img2.copy()
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    # print(ipm_matrix3*box)

                    # break
                    x = box[0]
                    # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    y =  box[1]
                    # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    w = box[2]
                    h = box[3]
                    
                    # print(x,y)
                    draw_bounding_box_base(img2, class_ids[i], confidences[i], round(x), round(y+170), round(x+w), round(y+h),classes,COLORS)
                    draw_bounding_box(img2_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                    f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")







                
                boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img3, cfg, weights,classes)


                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

          

            
                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./BB_Live_S3.txt"
                f = open(path,'w')
                # print("3",indices)
                # print("3",boxes)
                # print(ipm_matrix3)
                circle_coord = []

                # break
                img3_detect = img3.copy()
                for i in indices:
                    # i = i[0]
                    box = boxes[i]
                    # print(ipm_matrix3*box)

                    # break
                    x = box[0]
                    # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    y =  box[1]
                    # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    w = box[2]
                    h = box[3]
                    
                    # print(x,y)
                    draw_bounding_box_base(img3, class_ids[i], confidences[i], round(x), round(y+210), round(x+w), round(y+h),classes,COLORS)
                    draw_bounding_box(img3_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                    f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")


                ipm_mat = []

                # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]
                # i = 0
                # for pt in pts3:
                #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                #     i = i+1
                #     if i == len(colors):
                #         i = 0

                ## compute IPM matrix and apply it
                ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                ipm_mat.append(ipm1)
                # cv2.imwrite('img1_transformed.jpg',ipm1)



                # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                ipm_mat.append(ipm2)
                # cv2.imwrite('img2_transformed.jpg',ipm2)
                # ipm_pts3 = np.array([[501,452],[655,452], [662,350], [507,350]], dtype=np.float32)
                # ipm_matrix = cv2.getPerspectiveTransform(pts3, ipm_pts3)
                # ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix, top_view.shape[:2][::-1])
                # ipm_mat.append(ipm3)


                # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                ipm_mat.append(ipm3)
                # cv2.imwrite('img3_transformed.jpg',ipm3)


                
                # ipm_mat.append(ipm3)




                # print(ipm_mat)
                images_array = [img,img2,img3]

                # overlay = Image.open('img3_transformed.jpg').convert('RGBA')
                # image = Image.open('Top_View.jpeg').convert('RGBA')
                # image.paste(overlay, mask=overlay)
                # top_view_sol = image
                # image.save('result.png')




                h,w,n = top_view.shape
                top_view_sol = top_view.copy()
                top_view_sol = toRGBA(top_view_sol)
                # cv2.imshow("Top View Actual", top_view)
                # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                # (status, stitched) = stitcher.stitch(images_array)

                # cv2.imshow("stitched version", stitched)
                # ipm_mat = ipm_mat[0]

                # print(circle_coord[0][0])

                for ipm in ipm_mat:
                    # ipm = toRGBA(ipm)
                    # print("Here")
                    for i in range(h):
                      for j in range(w):
                        location_px = ipm[i][j].copy()
                        if location_px[3] == 1:
                             # and (top_view_sol[i][j]>0.7).all())
                          #  and (location_px <0.1).any():
                          top_view_sol[i][j] = location_px


                # im = Image.fromarray((top_view_sol * 255).astype(np.uint8))
                # im.save("./Prerecorded_Video_frames_Detection/Top_View"+str(ind)+".png")
                # im = cv2.imread(("./Prerecorded_Video_frames_Detection/Top_View"+str(ind)+".png"))
                # video.write(im)
                # ind = ind+1
                # break`


                # top_view_sol = cv2.cvtColor(top_view_sol, cv2.COLOR_RGBA2RGB)

                # rgbImage = cv2.cvtColor(npimage, cv2.COLOR_RGBA2RGB)


                # cv2.circle(top_view_sol, (int(circle_coord[0][0][0]),int(circle_coord[0][0][1])), 10, COLORS[0], thickness=10, lineType=8, shift=0)


                # plt.imshow(top_view_sol)
                # print(img.shape[:2][::-1])
                # cv2.imshow("top View",top_view_sol)
                # cv2.waitKey()

                # images_array.append(top_view_sol)
                # optputFile3.write(top_view_sol)

                # cv2.imwrite("Prerecorded_Video_Frames/TopView_PreRecorded_Recording_"+str(ind)+".png",top_view_sol)
                # ind = ind +1

                # display (or save) images
                cv2.imshow(windowName4, top_view_sol)
                cv2.imshow(windowName1, img_detect)
                cv2.imshow(windowName2, img2_detect)
                cv2.imshow(windowName3, img3_detect)
                # cv2.waitKey()
                # cv2.imshow('ipm', top_view_sol)
                if cv2.waitKey(1) == 27:
                    break

            # for imagee in images_array:
            #     optputFile3.write(imagee)
        # capture1.release()
        # capture3.release()
        # capture2.release()
        # cv2.destroyAllWindows()



            # cv2.imshow(windowName1, frame1)

            elif option_yolo == "s":
                # cap1 = cv2.VideoCapture('Recordings/Stream1Recording.avi')
                # cap2 = cv2.VideoCapture('Recordings/Stream2Recording.avi')
                # cap3 = cv2.VideoCapture('Recordings/Stream3Recording.avi')
                top_view = cv2.imread('Top_View.jpeg')
                size = (int(top_view.shape[1]),int(top_view.shape[0]))
                # optputFile3 = cv2.VideoWriter(
                    # 'TopView_PreRecorded_Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

                # windowName = "Sample Feed from Camera 1"
                # cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

                # windowName2 = "Sample Feed from Camera 2"
                # cv2.namedWindow(windowName2,cv2.WINDOW_NORMAL)

                # windowName3 = "Sample Feed from Camera 3"
                # cv2.namedWindow(windowName3,cv2.WINDOW_NORMAL)

                windowName4 = "Top View Of all Cameras"
                cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter('./Recordings/COVID_SOP_TopView.avi', fourcc, 1, size)

                # if (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                #     ret = True
                # else:
                #     ret = False

                ind = 0
                images_array = []
                ind_bb = 0

                px_img1 = (((1099-863) *2)/1.245)

                px_img2 = (((357-198) *2)/1.245)

                px_img3 = (((1297-1162) *2)/1.245)
                print(px_img1)
                print(px_img2)
                print(px_img3)
                colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]

                # cfg_person = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/darknet/cfg/yolov3.cfg"
                # weights_person = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/darknet/yolov3.weights"
                classes_person_file = "C:/Users/Hp/OneDrive/Documents/University/Semesters/Fall 2021/Computer Vision/Mini_Project_1/darknet/cfg/coco.data"

                with open(classes_person_file, 'r') as f:
                    classes_person = [line.strip() for line in f.readlines()]

                img = frame1
                img2 = frame2
                img3 = frame3
                top_view = cv2.imread('Top_View.jpeg')
                # img = cv2.imread("00000160.jpg")
                # img2 = cv2.imread("00000151.jpg")
                # img3 = cv2.imread("Stream1-2.jpg")
                # print(img3)

                th,tw,_ = top_view.shape
                # print(th,tw)

                h,w,n = img3.shape
                # img3 = img3[377:h][0:w]
                # cv2.imwrite("cap_1.jpg",img)
                # cv2.imwrite("cap_2.jpg",img2)
                # cv2.imwrite("cap_3.jpg",img3)
                # break


                # img = cv2.imread('img_h.jpeg')
                # img2 = cv2.imread('img_t.jpeg')
                # # img = cv2.imread('00000287.jpg')
                # img3 = cv2.imread('img_0.jpeg')
                    # img = img2



                # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)

                width = img.shape[1]
                height = img.shape[0]

                scale = 0.00392

                COLORS = np.random.uniform(0, 255, size=(len(classes_person), 3))
                # COLORS = colors
                

                net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                layer_names = net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
                # set input blob for the network
                net.setInput(blob)

                outs = net.forward(output_layers)
                # print(outs)
                


                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.7

                

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        
                        class_idx = np.argmax(scores)
                        # print(classval)
                        confidence = scores[class_idx]
                        # print(confidence)
                        if confidence > 0.5:

                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            # class_idx = np.argmax(scores)
                            if class_idx > (len(classes) - 1):
                                class_idx = class_idx % (len(classes)+1)
                            # print(classval)
                            if class_idx == 0:
                                class_ids.append(class_idx)
                                confidences.append(float(confidence))
                                boxes.append([x, y, w, h])

                # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                classes_covid = ["violation","no_violation"]
                # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                f = open(path,'a')
                # print(indices)
                # print(boxes)
                # print(ipm_matrix3)
                circle_coord = []
                toDraw = []
                # color_array = []
                count = [0, 0, 0]

                # break
                img_detect = img.copy()
                box = [boxes[i] for i in indices]
                # print(box)
                class_id_covid = []
                # if len(box) == 1:


                for i in range(len(box)-1):
                    if i + 1 != len(box):
                        dist = box[i][0] - box[i+1][0]
                        if dist > px_img1:
                            class_id_covid.append(1)
                        else:
                            class_id_covid.append(0)
                        x = box[i][0]
                        # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[i][1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = (box[i][2] + box[i+1][2])
                        h = (box[i][3])
                    else:
                        class_id_covid.append(1)

                    

                    # x = box[i][0]
                    # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    # y =  box[i][1]
                    # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    # w = box[i][2]
                    # h = box[i][3]
                    # draw_bounding_box_base(img, class_id_covid[i], confidences[i], round(x+100), round(y+200), round(x+w), round(y+h),classes_covid,COLORS)
                    toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                    # color_array.append(COLORS[class_ids[i]])
                    count[0] = count[0] + 1
                    draw_bounding_box(img_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+(w)), round(y+(h)),classes_covid,COLORS)



                net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                layer_names = net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                blob = cv2.dnn.blobFromImage(img2, scale, (416,416), (0,0,0), True, crop=False)
                # set input blob for the network
                net.setInput(blob)

                outs = net.forward(output_layers)
                # print(outs)
                


                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.95
                nms_threshold = 0.7

                

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        
                        class_idx = np.argmax(scores)
                        # print(class_idx)
                        # print(classval)
                        confidence = scores[class_idx]
                        # print(confidence)
                        if confidence > 0.5:

                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            # class_idx = np.argmax(scores)
                            if class_idx > (len(classes) - 1):
                                class_idx = class_idx % (len(classes)+1)
                            # print(classval)
                            if class_idx == 0:
                                class_ids.append(class_idx)
                                confidences.append(float(confidence))
                                boxes.append([x, y, w, h])


                # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img2, cfg, weights,classes)
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                # print(class_ids)

                classes_covid = ["violation","no_violation"]
                # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                f = open(path,'a')
                # print(indices)
                # print(boxes)
                # print(ipm_matrix3)
                circle_coord = []

                # break
                img2_detect = img2.copy()
                box = [boxes[i] for i in indices]
                # print(box)
                class_id_covid = []
                # if len(box) == 1:


                for i in range(len(box)-1):
                    if i + 1 != len(box):
                        dist = box[i][0] - box[i+1][0]
                        if dist > px_img2:
                            class_id_covid.append(1)
                        else:
                            class_id_covid.append(0)
                        x = box[i][0]
                        # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[i][1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = box[i][2] + box[i+1][2]
                        h = box[i][3] 
                        # h = h + 50
                        # y = y + 50
                        # x = x+ 50
                    else:
                        class_id_covid.append(1)

                    

                    # x = box[i][0]
                    # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    # y =  box[i][1]
                    # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    # w = box[i][2]
                    # h = box[i][3]
                    # draw_bounding_box_base(img2, class_id_covid[i], confidences[i], round(x+130), round(y+200), round(x+w), round(y+h),classes_covid,COLORS)
                    toDraw.append([round(x+130), round(y+200), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                    # color_array.append(COLORS[class_ids[i]])
                    count[1] = count[1] + 1
                    draw_bounding_box(img2_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes_covid,COLORS)


                # net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                # layer_names = net.getLayerNames()
                # output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                blob = cv2.dnn.blobFromImage(img3, scale, (416,416), (0,0,0), True, crop=False)
                # set input blob for the network
                net.setInput(blob)

                outs = net.forward(output_layers)
                # print(outs)
                


                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.7

                

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        
                        class_idx = np.argmax(scores)
                        # print(classval)
                        confidence = scores[class_idx]
                        # print(confidence)
                        if confidence > 0.5:

                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            # class_idx = np.argmax(scores)
                            if class_idx > (len(classes) - 1):
                                class_idx = class_idx % (len(classes)+1)
                            # print(classval)
                            if class_idx == 0:
                                class_ids.append(class_idx)
                                confidences.append(float(confidence))
                                boxes.append([x, y, w, h])


                # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img3, cfg, weights,classes)
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            # print(class_ids)


                classes_covid = ["violation","no_violation"]
                # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                f = open(path,'a')
                # print(indices)
                # print(boxes)
                # print(ipm_matrix3)
                circle_coord = []

                # break
                img3_detect = img3.copy()
                box = [boxes[i] for i in indices]
                # print(box)
                # print("confidence::", confidences)
                class_id_covid = []
                # if len(box) == 1:


                for i in range(len(box)-1):
                    if i + 1 != len(box):
                        dist = box[i][0] - box[i+1][0]
                        if dist > px_img2:
                            class_id_covid.append(1)
                        else:
                            class_id_covid.append(0)
                        x = box[i][0]
                        # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                        y =  box[i][1]
                        # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                        w = box[i][2] + box[i+1][2]
                        h = box[i][3]
                    else:
                        class_id_covid.append(1)

                    

                    # x = box[i][0]
                    # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                    # y =  box[i][1]
                    # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                    # w = box[i][2]
                    # h = box[i][3]
                    # draw_bounding_box_base(img3, class_id_covid[i], confidences[i], round(x+50), round(y+180), round(x+w), round(y+h),classes_covid,COLORS)
                    toDraw.append([ round(x+50), round(y+180), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                    # color_array.append(COLORS[class_ids[i]])
                    count[2] = count[2] + 1
                    draw_bounding_box(img3_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes_covid,COLORS)
                


                ipm_mat = []
                ind_bb = ind_bb +1

                # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]
                # i = 0
                # for pt in pts3:
                #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                #     i = i+1
                #     if i == len(colors):
                #         i = 0

                ## compute IPM matrix and apply it
                ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                ipm_mat.append(ipm1)
                # cv2.imwrite('img1_transformed.jpg',ipm1)



                # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                ipm_mat.append(ipm2)
              


                # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                ipm_mat.append(ipm3)
                # cv2.imwrite('img3_transformed.jpg',ipm3)


                
                # ipm_mat.append(ipm3)


                toDrawPts = []
                for i in range(0, count[0]):
                    #xpt = (toDraw[i][0] + toDraw[i][2])/2
                    #ypt = (toDraw[i][1] + toDraw[i][3])/2
                    #inter = np.matmul(ipm_matrix, np.array([round(xpt), round(ypt), 1]))
                    inter = np.matmul(ipm_matrix, np.array([toDraw[i][0], toDraw[i][1], 1]))
                    inter[0] = inter[0]/inter[2]
                    inter[1] = inter[1]/inter[2]
                    inter[2] = 1
                    toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                for i in range(count[0], count[0]+count[1]):
                    #xpt = (toDraw[i][0] + toDraw[i][2])/2
                    #ypt = (toDraw[i][1] + toDraw[i][3])/2
                    #inter = np.matmul(ipm_matrix2, np.array([round(xpt), round(ypt), 1]))
                    inter = np.matmul(ipm_matrix2, np.array([toDraw[i][0], toDraw[i][1], 1]))
                    inter[0] = inter[0]/inter[2]
                    inter[1] = inter[1]/inter[2]
                    inter[2] = 1
                    toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                for i in range(count[0]+count[1], count[0]+count[1]+count[2]):
                    #xpt = (toDraw[i][0] + toDraw[i][2])/2
                    #ypt = (toDraw[i][1] + toDraw[i][3])/2
                    #inter = np.matmul(ipm_matrix3, np.array([round(xpt), round(ypt), 1]))
                    inter = np.matmul(ipm_matrix3, np.array([toDraw[i][0], toDraw[i][1], 1]))
                    inter[0] = inter[0]/inter[2]
                    inter[1] = inter[1]/inter[2]
                    inter[2] = 1
                    toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])


                #print("SHAPE", toDrawPts[0].shape, toDrawPts[0][0], toDrawPts[0][1], toDrawPts[0][2])




                # print(ipm_mat)
                images_array = [img,img2,img3]

           




                h,w,n = top_view.shape
                top_view_sol = top_view.copy()
                top_view_sol = toRGBA(top_view_sol)
                top_view_sol_copy = top_view_sol.copy()
                # cv2.imshow("Top View Actual", top_view)
                # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                # (status, stitched) = stitcher.stitch(images_array)

                # cv2.imshow("stitched version", stitched)
                # ipm_mat = ipm_mat[0]

                # print(circle_coord[0][0])

                for ipm in ipm_mat:
                    # ipm = toRGBA(ipm)
                    # print("Here")
                    for i in range(h):
                      for j in range(w):
                        location_px = ipm[i][j].copy()
                        if location_px[3] == 1:
                             # and (top_view_sol[i][j]>0.7).all())
                          #  and (location_px <0.1).any():
                          top_view_sol[i][j] = location_px



                for i in range(len(toDrawPts)):
                    draw_bounding_box_base_pt(top_view_sol, toDrawPts[i][2], toDrawPts[i][0], toDrawPts[i][1], toDrawPts[i][3], toDrawPts[i][4])

                


                # im = Image.fromarray((top_view_sol * 255).astype(np.uint8))
                # im.save("./Prerecorded_Video_frames_Covid/Top_View"+str(ind)+".png")
                # im = cv2.imread(("./Prerecorded_Video_frames_Covid/Top_View"+str(ind)+".png"))
                # video.write(im)
                # ind = ind+1
                


               
                
                # display (or save) images
                cv2.imshow(windowName1, img_detect)
                cv2.imshow(windowName2, img2_detect)
                cv2.imshow(windowName3, img3_detect)
                cv2.imshow(windowName4, top_view_sol)
                # cv2.waitKey()
                # cv2.imshow('ipm', top_view_sol)
                if cv2.waitKey(1) == 27:
                    break

            # for imagee in images_array:
            #     optputFile3.write(imagee)
            
            

            
              

        #     cap1.release()
        #     # optputFile1.release()

        #     cap2.release()
        #     # optputFile2.release()

        #     cap3.release()
        # # optputFile3.release()
            # cv2.destroyAllWindows()
            elif option_yolo == "h":
                windowName4 = "Top View Of all Cameras"
                cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

                print("Press s for static heatmap\nPress a for animated heatmap \nPress c for COVID SOP Violated heatmap")
                option_heat = str(input())

                if option_heat == "s":
                   


                    # cap1 = cv2.VideoCapture('Recordings/Stream1Recording.avi')
                    # cap2 = cv2.VideoCapture('Recordings/Stream2Recording.avi')
                    # cap3 = cv2.VideoCapture('Recordings/Stream3Recording.avi')
                    top_view = cv2.imread('Top_View.jpeg')
                    size = (int(top_view.shape[1]),int(top_view.shape[0]))
                    # optputFile3 = cv2.VideoWriter(
                        # 'TopView_PreRecorded_Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

                    # windowName = "Sample Feed from Camera 1"
                    # cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

                    # windowName2 = "Sample Feed from Camera 2"
                    # cv2.namedWindow(windowName2,cv2.WINDOW_NORMAL)

                    # windowName3 = "Sample Feed from Camera 3"
                    # cv2.namedWindow(windowName3,cv2.WINDOW_NORMAL)

                    # windowName4 = "Top View Of all Cameras"
                    # cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

                    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # video = cv2.VideoWriter('./Recordings/Static_heatmap_PreRecorded.avi', fourcc, 1, size)

                    # if (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
                    #     ret = True
                    # else:
                    #     ret = False

                    ind = 0
                    images_array = []
                    heatmap_global = None
                    ind_bb = 0


                    while (capture1.isOpened() and capture2.isOpened() and capture3.isOpened()):
                        ret,img = capture1.read()
                        ret1,img2 = capture2.read()
                        ret2,img3 = capture3.read()
                    # img = frame1
                    # img2 = frame2
                    # img3 = frame3
                        top_view = cv2.imread('Top_View.jpeg')
                        # img = cv2.imread("00000160.jpg")
                        # img2 = cv2.imread("00000151.jpg")
                        # img3 = cv2.imread("Stream1-2.jpg")
                        # print(img3)

                        th,tw,_ = top_view.shape
                        # print(th,tw)

                        h,w,n = img3.shape
                        # img3 = img3[377:h][0:w]
                        # cv2.imwrite("capture_1.jpg",img)
                        # cv2.imwrite("capture_2.jpg",img2)
                        # cv2.imwrite("capture_3.jpg",img3)
                        # break


                    # img = cv2.imread('img_h.jpeg')
                    # img2 = cv2.imread('img_t.jpeg')
                    # # img = cv2.imread('00000287.jpg')
                    # img3 = cv2.imread('img_0.jpeg')
                        # img = img2

                        boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)


                        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                    # print(class_ids)

                    
                        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                        path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                        f = open(path,'a')
                        # print(indices)
                        # print(boxes)
                        # print(ipm_matrix3)
                        circle_coord = []

                        toDraw = []
                        count = [0, 0, 0]
                        # break
                        img_detect = img.copy()
                        for i in indices:
                            # i = i[0]
                            box = boxes[i]
                            # print(ipm_matrix3*box)

                            # break
                            x = box[0]
                            # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            y =  box[1]
                            # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            w = box[2]
                            h = box[3]
                            
                            # print(x,y)
                            #draw_bounding_box_base(img, class_ids[i], confidences[i], round(x), round(y+200), round(x+w), round(y+h),classes,COLORS)
                            toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                            count[0] = count[0] + 1
                            draw_bounding_box(img_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                            f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")

                            # cv2.circle(img, (round(x),round(y)), 10, COLORS[0], thickness=1, lineType=8, shift=0)
                            # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # print("\nDisplaying Detection Results...")
                        # windowName_detect = "Detection Result"
                        # cv2.namedWindow(windowName_detect)
                        # cv2.imshow(windowName_detect, img_detect)




                        boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img2, cfg, weights,classes)


                        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                    # print(class_ids)

                    
                        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                        path = "./Bounding_Boxes_Files/BB_prerecorded_S2"+str(ind_bb)+".txt"
                        f = open(path,'a')
                        # print(indices)
                        # print(boxes)
                        # print(ipm_matrix3)
                        circle_coord = []

                        # break
                        img2_detect = img2.copy()
                        for i in indices:
                            # i = i[0]
                            box = boxes[i]
                            # print(ipm_matrix3*box)

                            # break
                            x = box[0]
                            # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            y =  box[1]
                            # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            w = box[2]
                            h = box[3]
                            
                            # print(x,y)
                            #draw_bounding_box_base(img2, class_ids[i], confidences[i], round(x), round(y+170), round(x+w), round(y+h),classes,COLORS)
                            toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                            count[1] = count[1] + 1
                            draw_bounding_box(img2_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                            f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")







                        
                        boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img3, cfg, weights,classes)


                        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                    # print(class_ids)

                    
                        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                        path = "./Bounding_Boxes_Files/BB_prerecorded_S3"+str(ind_bb)+".txt"
                        f = open(path,'a')
                        # print(indices)
                        # print(boxes)
                        # print(ipm_matrix3)
                        circle_coord = []

                        # break
                        img3_detect = img3.copy()
                        for i in indices:
                            # i = i[0]
                            box = boxes[i]
                            # print(ipm_matrix3*box)

                            # break
                            x = box[0]
                            # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            y =  box[1]
                            # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            w = box[2]
                            h = box[3]
                            
                            # print(x,y)
                            #draw_bounding_box_base(img3, class_ids[i], confidences[i], round(x), round(y+120), round(x+w), round(y+h),classes,COLORS)
                            toDraw.append([round(x), round(y+120), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                            count[2] = count[2] + 1
                            draw_bounding_box(img3_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                            f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")


                        ipm_mat = []
                        ind_bb = ind_bb +1

                        # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                        # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                        # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                        # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                        # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                        pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                        pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                        pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                        pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                        # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                        pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                        # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                        # colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]
                        # i = 0
                        # for pt in pts3:
                        #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                        #     i = i+1
                        #     if i == len(colors):
                        #         i = 0

                        ## compute IPM matrix and apply it
                        ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                        # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                        # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                        ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                        # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                        ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                        # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                        ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                        ipm_mat.append(ipm1)
                        # cv2.imwrite('img1_transformed.jpg',ipm1)



                        # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                        # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                        ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                        ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                        # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                        ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                        ipm_mat.append(ipm2)
                        # cv2.imwrite('img2_transformed.jpg',ipm2)
                        # ipm_pts3 = np.array([[501,452],[655,452], [662,350], [507,350]], dtype=np.float32)
                        # ipm_matrix = cv2.getPerspectiveTransform(pts3, ipm_pts3)
                        # ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix, top_view.shape[:2][::-1])
                        # ipm_mat.append(ipm3)


                        # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                        ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                        # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                        ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                        # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                        ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                        ipm_mat.append(ipm3)
                        # cv2.imwrite('img3_transformed.jpg',ipm3)


                        toDrawPts = []
                        for i in range(0, count[0]):
                            #xpt = (toDraw[i][0] + toDraw[i][2])/2
                            #ypt = (toDraw[i][1] + toDraw[i][3])/2
                            #inter = np.matmul(ipm_matrix, np.array([round(xpt), round(ypt), 1]))
                            inter = np.matmul(ipm_matrix, np.array([toDraw[i][0], toDraw[i][1], 1]))
                            inter[0] = inter[0]/inter[2]
                            inter[1] = inter[1]/inter[2]
                            inter[2] = 1
                            toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                        for i in range(count[0], count[0]+count[1]):
                            #xpt = (toDraw[i][0] + toDraw[i][2])/2
                            #ypt = (toDraw[i][1] + toDraw[i][3])/2
                            #inter = np.matmul(ipm_matrix2, np.array([round(xpt), round(ypt), 1]))
                            inter = np.matmul(ipm_matrix2, np.array([toDraw[i][0], toDraw[i][1], 1]))
                            inter[0] = inter[0]/inter[2]
                            inter[1] = inter[1]/inter[2]
                            inter[2] = 1
                            toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                        for i in range(count[0]+count[1], count[0]+count[1]+count[2]):
                            #xpt = (toDraw[i][0] + toDraw[i][2])/2
                            #ypt = (toDraw[i][1] + toDraw[i][3])/2
                            #inter = np.matmul(ipm_matrix3, np.array([round(xpt), round(ypt), 1]))
                            inter = np.matmul(ipm_matrix3, np.array([toDraw[i][0], toDraw[i][1], 1]))
                            inter[0] = inter[0]/inter[2]
                            inter[1] = inter[1]/inter[2]
                            inter[2] = 1
                            toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])


                        #print("SHAPE", toDrawPts[0].shape, toDrawPts[0][0], toDrawPts[0][1], toDrawPts[0][2])
                        # ipm_mat.append(ipm3)




                        # print(ipm_mat)
                        images_array = [img,img2,img3]

                        # overlay = Image.open('img3_transformed.jpg').convert('RGBA')
                        # image = Image.open('Top_View.jpeg').convert('RGBA')
                        # image.paste(overlay, mask=overlay)
                        # top_view_sol = image
                        # image.save('result.png')




                        top_view = cv2.cvtColor(cv2.imread('Top_View.jpeg'),cv2.COLOR_BGR2RGB)/255
                        h,w,n = top_view.shape
                        top_view_sol = top_view.copy()
                        top_view_sol = toRGBA(top_view_sol)
                        # cv2.imshow("Top View Actual", top_view)
                        # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                        # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                        # (status, stitched) = stitcher.stitch(images_array)

                        # cv2.imshow("stitched version", stitched)
                        # ipm_mat = ipm_mat[0]

                        # print(circle_coord[0][0])

                        for ipm in ipm_mat:
                            # ipm = toRGBA(ipm)
                            # print("Here")
                            for i in range(h):
                              for j in range(w):
                                location_px = ipm[i][j].copy()
                                if location_px[3] == 1:
                                     # and (top_view_sol[i][j]>0.7).all())
                                  #  and (location_px <0.1).any():
                                  top_view_sol[i][j] = location_px


                        for i in range(len(toDrawPts)):
                            draw_bounding_box_base_pt(top_view_sol, toDrawPts[i][2], toDrawPts[i][0], toDrawPts[i][1], toDrawPts[i][3], toDrawPts[i][4])

                                       # cv2.cvtColor(top_view_sol,cv2.COLOR_BGR2RGB)/255


                        


                        k = 21
                            # x=0
                        gauss = cv2.getGaussianKernel(k,np.sqrt(64))
                        gauss = gauss * gauss.T
                        gauss = (gauss/gauss[int(k/2),int(k/2)])

                        heatmap = np.zeros((top_view.shape[0],top_view.shape[1],3)).astype(np.float32)
                        # heatmap = toRGBA(heatmap)
                        # heatmap[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])] += 1
                        # heatmap[heatmap > 0.8] = 0


                        j = cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),cv2.COLORMAP_SUMMER),cv2.COLOR_BGR2RGB).astype(np.float32)/255
                        points = toDrawPts

                        for p in points:
                            # print(p[0]- int(k/2))
                            # print(p[0]+int(k/2)+1)
                            # print(p[1]- int(k/2))
                            # print(p[1]+int(k/2)+1)

                            # print(heatmap[int(p[0]):int(p[0]+int(k/2)+1),int(p[1]-int(k/2)):int(p[1]+int(k/2)+1),:].shape)
                            # print(j.shape)



                            try:
                                # b = heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:]
                                # c = j + b

                                # heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:] = j
                                draw_bounding_box_base_pt(heatmap, p[2], p[0], p[1], p[3], p[4])
                            except:
                                pass

                        # m = np.max(heatmap, axis = (0,1))
                        # heatmap = heatmap/m




                        heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)

                        mask = np.where(heatmap > 0.2,1,0).astype(np.float32)
                        inv_mask = np.ones((top_view.shape[0],top_view.shape[1],3))*(1 - mask)[:,:,None]
                        
                        im = Image.fromarray((heatmap * 255).astype(np.uint8))
                        im.save('./frames_S1/frames'+str(ind)+".png")


                        images = cv2.imread('./frames_S1/frames'+str(ind)+".png", 0)
                        # colormap = plt.get_cmap('inferno')
                        # heatmap = (colormap(images) * 2**16).astype(np.uint16)[:,:,:3]
                        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

                        # heatmap = (heatmap/255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(images, cv2.COLORMAP_AUTUMN)
                        # try:
                        #     heatmap_global == None:
                        #     heatmap_global = heatmap
                        # else:
                        try:
                            heatmap_global = heatmap_global + heatmap
                        except:
                            heatmap_global = heatmap
                        # heatmap_array.append(heatmap)
                        # # imask = (inv_mask/255).astype(np.uint8)
                        # # imask = cv2.applyColorMap(imask, cv2.COLORMAP_HOT)

                        
                        new_top = np.where(inv_mask>0,top_view,inv_mask)
                        new_top = inv_mask * top_view

                        new_top = new_top + heatmap_global





                        # cv2.imshow("New_Top_View",new_top)
                        # cv2.imshow("Heatmap",heatmap)
                        # cv2.imshow("Top View Sol",top_view_sol)
                        # cv2.imshow("Image",img_detect)



                        # cv2.imshow("Actual img",img)

                        # cv2.waitKey()
                        

                        

                        # im = Image.fromarray((new_top * 255).astype(np.uint8))
                        # im.save("./Prerecorded_Video_Heatmap_static/Top_View"+str(ind)+".png")
                        # im = cv2.imread(("./Prerecorded_Video_Heatmap_static/Top_View"+str(ind)+".png"))
                        # video.write(im)
                        # ind = ind+1
                        ind = ind + 1
                        # break`


                        # top_view_sol = cv2.cvtColor(top_view_sol, cv2.COLOR_RGBA2RGB)

                        # rgbImage = cv2.cvtColor(npimage, cv2.COLOR_RGBA2RGB)


                        # cv2.circle(top_view_sol, (int(circle_coord[0][0][0]),int(circle_coord[0][0][1])), 10, COLORS[0], thickness=10, lineType=8, shift=0)


                        # plt.imshow(top_view_sol)
                        # print(img.shape[:2][::-1])
                        # cv2.imshow("top View",top_view_sol)
                        # cv2.waitKey()

                        # images_array.append(top_view_sol)
                        # optputFile3.write(top_view_sol)

                        # cv2.imwrite("Prerecorded_Video_Frames/TopView_PreRecorded_Recording_"+str(ind)+".png",top_view_sol)
                        # ind = ind +1

                        # display (or save) images
                        cv2.imshow(windowName4, new_top)
                        cv2.imshow(windowName1, img_detect)
                        cv2.imshow(windowName2, img2_detect)
                        cv2.imshow(windowName3, img3_detect)
                        # cv2.waitKey()
                        # cv2.imshow('ipm', top_view_sol)
                        if cv2.waitKey(1) == 27:
                            break

                    # for imagee in images_array:
                    #     optputFile3.write(imagee)
                    
                    

                    # imgs_list = glob.glob("./Prerecorded_Video_Heatmap_static/*.png")
                    # for imgs in imgs_list:
                    #     img = cv2.imread(imgs)
                    #     video.write(img)

                      

                #     capture1.release()
                #     # optputFile1.release()

                #     capture2.release()
                #     # optputFile2.release()

                #     capture3.release()
                # # optputFile3.release()
                #     cv2.destroyAllWindows()

                if option_heat == "a":



                    # capture1 = cv2.VideoCapture('Recordings/Stream1Recording.avi')
                    # capture2 = cv2.VideoCapture('Recordings/Stream2Recording.avi')
                    # capture3 = cv2.VideoCapture('Recordings/Stream3Recording.avi')
                    top_view = cv2.imread('Top_View.jpeg')
                    size = (int(top_view.shape[1]),int(top_view.shape[0]))
                    # optputFile3 = cv2.VideoWriter(
                        # 'TopView_PreRecorded_Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

                    # windowName = "Sample Feed from Camera 1"
                    # cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

                    # windowName2 = "Sample Feed from Camera 2"
                    # cv2.namedWindow(windowName2,cv2.WINDOW_NORMAL)

                    # windowName3 = "Sample Feed from Camera 3"
                    # cv2.namedWindow(windowName3,cv2.WINDOW_NORMAL)

                    # windowName4 = "Top View Of all Cameras"
                    # cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

                    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # video = cv2.VideoWriter('./Recordings/Animated_Heatmpa_Prerecorded.avi', fourcc, 1, size)

                    # if (capture1.isOpened() and capture2.isOpened() and capture3.isOpened()):
                    #     ret = True
                    # else:
                    #     ret = False

                    ind = 0
                    images_array = []
                    heatmap_array = []
                    frames_to_consider = 5

                    ind_bb = 0


                    while (capture1.isOpened() and capture2.isOpened() and capture3.isOpened()):
                        ret,img = capture1.read()
                        ret1,img2 = capture2.read()
                        ret2,img3 = capture3.read()
                    # img = frame1
                    # img2 = frame2
                    # img3 = frame3
                        top_view = cv2.imread('Top_View.jpeg')
                            # img = cv2.imread("00000160.jpg")
                            # img2 = cv2.imread("00000151.jpg")
                            # img3 = cv2.imread("Stream1-2.jpg")
                            # print(img3)

                        th,tw,_ = top_view.shape
                        # print(th,tw)

                        h,w,n = img3.shape
                        # img3 = img3[377:h][0:w]
                        # cv2.imwrite("capture_1.jpg",img)
                        # cv2.imwrite("capture_2.jpg",img2)
                        # cv2.imwrite("capture_3.jpg",img3)
                        # break


                    # img = cv2.imread('img_h.jpeg')
                    # img2 = cv2.imread('img_t.jpeg')
                    # # img = cv2.imread('00000287.jpg')
                    # img3 = cv2.imread('img_0.jpeg')
                        # img = img2

                        boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)


                        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                        # print(class_ids)

                        
                        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                        path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                        f = open(path,'a')
                        # print(indices)
                        # print(boxes)
                        # print(ipm_matrix3)
                        circle_coord = []

                        toDraw = []
                        count = [0, 0, 0]
                        # break
                        img_detect = img.copy()
                        for i in indices:
                            # i = i[0]
                            box = boxes[i]
                            # print(ipm_matrix3*box)

                            # break
                            x = box[0]
                            # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            y =  box[1]
                            # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            w = box[2]
                            h = box[3]
                            
                            # print(x,y)
                            #draw_bounding_box_base(img, class_ids[i], confidences[i], round(x), round(y+200), round(x+w), round(y+h),classes,COLORS)
                            toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                            count[0] = count[0] + 1
                            draw_bounding_box(img_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                            f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")

                            # cv2.circle(img, (round(x),round(y)), 10, COLORS[0], thickness=1, lineType=8, shift=0)
                            # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # print("\nDisplaying Detection Results...")
                        # windowName_detect = "Detection Result"
                        # cv2.namedWindow(windowName_detect)
                        # cv2.imshow(windowName_detect, img_detect)




                        boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img2, cfg, weights,classes)


                        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                    # print(class_ids)

                    
                        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                        path = "./Bounding_Boxes_Files/BB_prerecorded_S2"+str(ind_bb)+".txt"
                        f = open(path,'a')
                        # print(indices)
                        # print(boxes)
                        # print(ipm_matrix3)
                        circle_coord = []

                        # break
                        img2_detect = img2.copy()
                        for i in indices:
                            # i = i[0]
                            box = boxes[i]
                            # print(ipm_matrix3*box)

                            # break
                            x = box[0]
                            # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            y =  box[1]
                            # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            w = box[2]
                            h = box[3]
                            
                            # print(x,y)
                            #draw_bounding_box_base(img2, class_ids[i], confidences[i], round(x), round(y+170), round(x+w), round(y+h),classes,COLORS)
                            toDraw.append([round(x), round(y+200), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                            count[1] = count[1] + 1
                            draw_bounding_box(img2_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                            f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")







                        
                        boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img3, cfg, weights,classes)


                        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                    # print(class_ids)

                    
                        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                        path = "./Bounding_Boxes_Files/BB_prerecorded_S3"+str(ind_bb)+".txt"
                        f = open(path,'a')
                        # print(indices)
                        # print(boxes)
                        # print(ipm_matrix3)
                        circle_coord = []

                        # break
                        img3_detect = img3.copy()
                        for i in indices:
                            # i = i[0]
                            box = boxes[i]
                            # print(ipm_matrix3*box)

                            # break
                            x = box[0]
                            # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            y =  box[1]
                            # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            w = box[2]
                            h = box[3]
                            
                            # print(x,y)
                            #draw_bounding_box_base(img3, class_ids[i], confidences[i], round(x), round(y+120), round(x+w), round(y+h),classes,COLORS)
                            toDraw.append([round(x), round(y+120), round(x+w), round(y+h), class_ids[i], classes, colors[class_ids[i]]])
                            count[2] = count[2] + 1
                            draw_bounding_box(img3_detect, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
                            f.write(str(classes[class_ids[i]])+" ("+str(round(x))+","+ str(round(y))+") ("+str( round(x+w))+","+str(round(y+h))+")\n")


                        ipm_mat = []
                        ind_bb = ind_bb +1

                        # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                        # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                        # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                        # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                        # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                        pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                        pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                        pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                        pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                        # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                        pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                        # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                        # colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]
                        # i = 0
                        # for pt in pts3:
                        #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                        #     i = i+1
                        #     if i == len(colors):
                        #         i = 0

                        ## compute IPM matrix and apply it
                        ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                        # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                        # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                        ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                        # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                        ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                        # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                        ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                        ipm_mat.append(ipm1)
                        # cv2.imwrite('img1_transformed.jpg',ipm1)



                        # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                        # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                        ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                        ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                        # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                        ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                        ipm_mat.append(ipm2)
                        # cv2.imwrite('img2_transformed.jpg',ipm2)
                        # ipm_pts3 = np.array([[501,452],[655,452], [662,350], [507,350]], dtype=np.float32)
                        # ipm_matrix = cv2.getPerspectiveTransform(pts3, ipm_pts3)
                        # ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix, top_view.shape[:2][::-1])
                        # ipm_mat.append(ipm3)


                        # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                        ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                        # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                        ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                        # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                        ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                        ipm_mat.append(ipm3)
                        # cv2.imwrite('img3_transformed.jpg',ipm3)


                        toDrawPts = []
                        for i in range(0, count[0]):
                            #xpt = (toDraw[i][0] + toDraw[i][2])/2
                            #ypt = (toDraw[i][1] + toDraw[i][3])/2
                            #inter = np.matmul(ipm_matrix, np.array([round(xpt), round(ypt), 1]))
                            inter = np.matmul(ipm_matrix, np.array([toDraw[i][0], toDraw[i][1], 1]))
                            inter[0] = inter[0]/inter[2]
                            inter[1] = inter[1]/inter[2]
                            inter[2] = 1
                            toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                        for i in range(count[0], count[0]+count[1]):
                            #xpt = (toDraw[i][0] + toDraw[i][2])/2
                            #ypt = (toDraw[i][1] + toDraw[i][3])/2
                            #inter = np.matmul(ipm_matrix2, np.array([round(xpt), round(ypt), 1]))
                            inter = np.matmul(ipm_matrix2, np.array([toDraw[i][0], toDraw[i][1], 1]))
                            inter[0] = inter[0]/inter[2]
                            inter[1] = inter[1]/inter[2]
                            inter[2] = 1
                            toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                        for i in range(count[0]+count[1], count[0]+count[1]+count[2]):
                            #xpt = (toDraw[i][0] + toDraw[i][2])/2
                            #ypt = (toDraw[i][1] + toDraw[i][3])/2
                            #inter = np.matmul(ipm_matrix3, np.array([round(xpt), round(ypt), 1]))
                            inter = np.matmul(ipm_matrix3, np.array([toDraw[i][0], toDraw[i][1], 1]))
                            inter[0] = inter[0]/inter[2]
                            inter[1] = inter[1]/inter[2]
                            inter[2] = 1
                            toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])


                        #print("SHAPE", toDrawPts[0].shape, toDrawPts[0][0], toDrawPts[0][1], toDrawPts[0][2])
                        # ipm_mat.append(ipm3)




                        # print(ipm_mat)
                        images_array = [img,img2,img3]

                        # overlay = Image.open('img3_transformed.jpg').convert('RGBA')
                        # image = Image.open('Top_View.jpeg').convert('RGBA')
                        # image.paste(overlay, mask=overlay)
                        # top_view_sol = image
                        # image.save('result.png')




                        top_view = cv2.cvtColor(cv2.imread('Top_View.jpeg'),cv2.COLOR_BGR2RGB)/255
                        h,w,n = top_view.shape
                        top_view_sol = top_view.copy()
                        top_view_sol = toRGBA(top_view_sol)
                        # cv2.imshow("Top View Actual", top_view)
                        # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                        # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                        # (status, stitched) = stitcher.stitch(images_array)

                        # cv2.imshow("stitched version", stitched)
                        # ipm_mat = ipm_mat[0]

                        # print(circle_coord[0][0])

                        for ipm in ipm_mat:
                            # ipm = toRGBA(ipm)
                            # print("Here")
                            for i in range(h):
                              for j in range(w):
                                location_px = ipm[i][j].copy()
                                if location_px[3] == 1:
                                     # and (top_view_sol[i][j]>0.7).all())
                                  #  and (location_px <0.1).any():
                                  top_view_sol[i][j] = location_px


                        for i in range(len(toDrawPts)):
                            draw_bounding_box_base_pt(top_view_sol, toDrawPts[i][2], toDrawPts[i][0], toDrawPts[i][1], toDrawPts[i][3], toDrawPts[i][4])

                                       # cv2.cvtColor(top_view_sol,cv2.COLOR_BGR2RGB)/255


                        


                        k = 21
                            # x=0
                        gauss = cv2.getGaussianKernel(k,np.sqrt(64))
                        gauss = gauss * gauss.T
                        gauss = (gauss/gauss[int(k/2),int(k/2)])

                        heatmap = np.zeros((top_view.shape[0],top_view.shape[1],3)).astype(np.float32)
                        # heatmap = toRGBA(heatmap)
                        # heatmap[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])] += 1
                        # heatmap[heatmap > 0.8] = 0


                        j = cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),cv2.COLORMAP_SUMMER),cv2.COLOR_BGR2RGB).astype(np.float32)/255
                        points = toDrawPts

                        for p in points:
                            # print(p[0]- int(k/2))
                            # print(p[0]+int(k/2)+1)
                            # print(p[1]- int(k/2))
                            # print(p[1]+int(k/2)+1)

                            # print(heatmap[int(p[0]):int(p[0]+int(k/2)+1),int(p[1]-int(k/2)):int(p[1]+int(k/2)+1),:].shape)
                            # print(j.shape)



                            try:
                                # b = heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:]
                                # c = j + b

                                # heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:] = j
                                draw_bounding_box_base_pt(heatmap, p[2], p[0], p[1], p[3], p[4])
                            except:
                                pass

                        # m = np.max(heatmap, axis = (0,1))
                        # heatmap = heatmap/m




                        heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)

                        mask = np.where(heatmap > 0.2,1,0).astype(np.float32)
                        inv_mask = np.ones((top_view.shape[0],top_view.shape[1],3))*(1 - mask)[:,:,None]
                        
                        im = Image.fromarray((heatmap * 255).astype(np.uint8))
                        im.save('./frames_S1/frames'+str(ind)+".png")


                        images = cv2.imread('./frames_S1/frames'+str(ind)+".png", 0)
                        # colormap = plt.get_cmap('inferno')
                        # heatmap = (colormap(images) * 2**16).astype(np.uint16)[:,:,:3]
                        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

                        # heatmap = (heatmap/255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(images, cv2.COLORMAP_AUTUMN)
                        heatmap_array.append(heatmap)
                        # # imask = (inv_mask/255).astype(np.uint8)
                        # # imask = cv2.applyColorMap(imask, cv2.COLORMAP_HOT)
                        top_view = rgba2rgb(top_view)

                        
                        new_top = np.where(inv_mask>0,top_view,inv_mask)
                        new_top = inv_mask * top_view

                        try:
                            heatmap_new = heatmap_array[len(heatmap_array) - frames_to_consider]
                            for i in range(len(heatmap_array) - frames_to_consider+1,len(heatmap_array)):
                                heatmap_new = heatmap_new+heatmap_array[i]
                        except:
                            heatmap_new = heatmap_array[0]
                            for i in range(1,len(heatmap_array)):
                                heatmap_new = heatmap_new+heatmap_array[i]


                        new_top = new_top + heatmap_new





                        # cv2.imshow("New_Top_View",new_top)
                        # cv2.imshow("Heatmap",heatmap)
                        # cv2.imshow("Top View Sol",top_view_sol)
                        # cv2.imshow("Image",img_detect)



                        # cv2.imshow("Actual img",img)

                        # cv2.waitKey()
                        # ind = ind + 1

                        

                        # im = Image.fromarray((top_view_sol * 255).astype(np.uint8))
                        # im.save("./Prerecorded_Video_Heatmap_Animated/Top_View"+str(ind)+".png")
                        # im = cv2.imread(("./Prerecorded_Video_Heatmap_Animated/Top_View"+str(ind)+".png"))
                        # video.write(im)
                        ind = ind+1
                        # break`


                        # top_view_sol = cv2.cvtColor(top_view_sol, cv2.COLOR_RGBA2RGB)

                        # rgbImage = cv2.cvtColor(npimage, cv2.COLOR_RGBA2RGB)


                        # cv2.circle(top_view_sol, (int(circle_coord[0][0][0]),int(circle_coord[0][0][1])), 10, COLORS[0], thickness=10, lineType=8, shift=0)


                        # plt.imshow(top_view_sol)
                        # print(img.shape[:2][::-1])
                        # cv2.imshow("top View",top_view_sol)
                        # cv2.waitKey()

                        # images_array.append(top_view_sol)
                        # optputFile3.write(top_view_sol)

                        # cv2.imwrite("Prerecorded_Video_Frames/TopView_PreRecorded_Recording_"+str(ind)+".png",top_view_sol)
                        # ind = ind +1

                        # display (or save) images
                        cv2.imshow(windowName4, new_top)
                        cv2.imshow(windowName1, img_detect)
                        cv2.imshow(windowName2, img2_detect)
                        cv2.imshow(windowName3, img3_detect)
                        # cv2.waitKey()
                        # cv2.imshow('ipm', top_view_sol)
                        if cv2.waitKey(1) == 27:
                            break

                # for imagee in images_array:
                #     optputFile3.write(imagee)
                
                

            #     imgs_list = glob.glob("./Prerecorded_Video_Frames/*.png")
            #     for imgs in imgs_list:
            #         img = cv2.imread(imgs)
            #         video.write(img)

                  

            #     capture1.release()
            #     # optputFile1.release()

            #     capture2.release()
            #     # optputFile2.release()

            #     capture3.release()
            # # optputFile3.release()
            #     cv2.destroyAllWindows()

                if option_heat == "c":
                # capture1 = cv2.VideoCapture('Recordings/Stream1Recording.avi')
                # capture2 = cv2.VideoCapture('Recordings/Stream2Recording.avi')
                # capture3 = cv2.VideoCapture('Recordings/Stream3Recording.avi')
                    top_view = cv2.imread('Top_View.jpeg')
                    size = (int(top_view.shape[1]),int(top_view.shape[0]))
                    # optputFile3 = cv2.VideoWriter(
                        # 'TopView_PreRecorded_Recording.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

                    # windowName = "Sample Feed from Camera 1"
                    # cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

                    # windowName2 = "Sample Feed from Camera 2"
                    # cv2.namedWindow(windowName2,cv2.WINDOW_NORMAL)

                    # windowName3 = "Sample Feed from Camera 3"
                    # cv2.namedWindow(windowName3,cv2.WINDOW_NORMAL)

                    # windowName4 = "Top View Of all Cameras"
                    # cv2.namedWindow(windowName4, cv2.WINDOW_NORMAL)

                    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # video = cv2.VideoWriter('./Recordings/COVID_Heatmap_Prerecorded.avi', fourcc, 1, size)

                    # if (capture1.isOpened() and capture2.isOpened() and capture3.isOpened()):
                    #     ret = True
                    # else:
                    #     ret = False

                    ind = 0
                    images_array = []
                    heatmap_array = []
                    ind_bb = 0
                    frames_to_consider = 5

                    px_img1 = (((1099-863) *2)/1.245)

                    px_img2 = (((357-198) *2)/1.245)

                    px_img3 = (((1297-1162) *2)/1.245)


                    while (capture1.isOpened() and capture2.isOpened() and capture3.isOpened()):
                        ret,img = capture1.read()
                        ret1,img2 = capture2.read()
                        ret2,img3 = capture3.read()
                    # img = frame1
                    # img2 = frame2
                    # img3 = frame3
                        top_view = cv2.imread('Top_View.jpeg')
                        # img = cv2.imread("00000160.jpg")
                        # img2 = cv2.imread("00000151.jpg")
                        # img3 = cv2.imread("Stream1-2.jpg")
                        # print(img3)

                        th,tw,_ = top_view.shape
                        # print(th,tw)

                        h,w,n = img3.shape
                        # img3 = img3[377:h][0:w]
                        # cv2.imwrite("cap_1.jpg",img)
                        # cv2.imwrite("cap_2.jpg",img2)
                        # cv2.imwrite("cap_3.jpg",img3)
                        # break


                        # img = cv2.imread('img_h.jpeg')
                        # img2 = cv2.imread('img_t.jpeg')
                        # # img = cv2.imread('00000287.jpg')
                        # img3 = cv2.imread('img_0.jpeg')
                            # img = img2



                        # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)

                        width = img.shape[1]
                        height = img.shape[0]

                        scale = 0.00392

                        COLORS = np.random.uniform(0, 255, size=(len(classes_person), 3))
                        # COLORS = colors
                        

                        net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                        layer_names = net.getLayerNames()
                        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                        blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
                        # set input blob for the network
                        net.setInput(blob)

                        outs = net.forward(output_layers)
                        # print(outs)
                        


                        class_ids = []
                        confidences = []
                        boxes = []
                        conf_threshold = 0.5
                        nms_threshold = 0.7

                        

                        for out in outs:
                            for detection in out:
                                scores = detection[5:]
                                
                                class_idx = np.argmax(scores)
                                # print(classval)
                                confidence = scores[class_idx]
                                # print(confidence)
                                if confidence > 0.5:

                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)
                                    x = center_x - w / 2
                                    y = center_y - h / 2
                                    # class_idx = np.argmax(scores)
                                    if class_idx > (len(classes) - 1):
                                        class_idx = class_idx % (len(classes)+1)
                                    # print(classval)
                                    if class_idx == 0:
                                        class_ids.append(class_idx)
                                        confidences.append(float(confidence))
                                        boxes.append([x, y, w, h])

                        # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img, cfg, weights,classes)
                        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                        # print(class_ids)

                        classes_covid = ["violation","no_violation"]
                        # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                        path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                        f = open(path,'a')
                        # print(indices)
                        # print(boxes)
                        # print(ipm_matrix3)
                        circle_coord = []
                        toDraw = []
                        # color_array = []
                        count = [0, 0, 0]

                        # break
                        img_detect = img.copy()
                        box = [boxes[i] for i in indices]
                        # print(box)
                        class_id_covid = []
                        # if len(box) == 1:


                        for i in range(len(box)-1):
                            if i + 1 != len(box):
                                dist = box[i][0] - box[i+1][0]
                                if dist > px_img1:
                                    class_id_covid.append(1)
                                else:
                                    class_id_covid.append(0)
                                x = box[i][0]
                                # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                                y =  box[i][1]
                                # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                                w = (box[i][2] + box[i+1][2])
                                h = (box[i][3])
                            else:
                                class_id_covid.append(1)

                            

                            # x = box[i][0]
                            # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            # y =  box[i][1]
                            # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            # w = box[i][2]
                            # h = box[i][3]
                            # draw_bounding_box_base(img, class_id_covid[i], confidences[i], round(x+100), round(y+200), round(x+w), round(y+h),classes_covid,COLORS)
                            toDraw.append([round(x+150), round(y+200), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                            # color_array.append(COLORS[class_ids[i]])
                            count[0] = count[0] + 1
                            draw_bounding_box(img_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+(w)), round(y+(h)),classes_covid,COLORS)



                        net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                        layer_names = net.getLayerNames()
                        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                        blob = cv2.dnn.blobFromImage(img2, scale, (416,416), (0,0,0), True, crop=False)
                        # set input blob for the network
                        net.setInput(blob)

                        outs = net.forward(output_layers)
                        # print(outs)
                        


                        class_ids = []
                        confidences = []
                        boxes = []
                        conf_threshold = 0.95
                        nms_threshold = 0.7

                        

                        for out in outs:
                            for detection in out:
                                scores = detection[5:]
                                
                                class_idx = np.argmax(scores)
                                # print(class_idx)
                                # print(classval)
                                confidence = scores[class_idx]
                                # print(confidence)
                                if confidence > 0.5:

                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)
                                    x = center_x - w / 2
                                    y = center_y - h / 2
                                    # class_idx = np.argmax(scores)
                                    if class_idx > (len(classes) - 1):
                                        class_idx = class_idx % (len(classes)+1)
                                    # print(classval)
                                    if class_idx == 0:
                                        class_ids.append(class_idx)
                                        confidences.append(float(confidence))
                                        boxes.append([x, y, w, h])


                        # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img2, cfg, weights,classes)
                        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                    # print(class_ids)

                        classes_covid = ["violation","no_violation"]
                        # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                        path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                        f = open(path,'a')
                        # print(indices)
                        # print(boxes)
                        # print(ipm_matrix3)
                        circle_coord = []

                        # break
                        img2_detect = img2.copy()
                        box = [boxes[i] for i in indices]
                        # print(box)
                        class_id_covid = []
                        # if len(box) == 1:


                        for i in range(len(box)-1):
                            if i + 1 != len(box):
                                dist = box[i][0] - box[i+1][0]
                                if dist > px_img2:
                                    class_id_covid.append(1)
                                else:
                                    class_id_covid.append(0)
                                x = box[i][0]
                                # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                                y =  box[i][1]
                                # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                                w = box[i][2] + box[i+1][2]
                                h = box[i][3] 
                                # h = h + 50
                                # y = y + 50
                                # x = x+ 50
                            else:
                                class_id_covid.append(1)

                            

                            # x = box[i][0]
                            # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            # y =  box[i][1]
                            # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            # w = box[i][2]
                            # h = box[i][3]
                            # draw_bounding_box_base(img2, class_id_covid[i], confidences[i], round(x+130), round(y+200), round(x+w), round(y+h),classes_covid,COLORS)
                            toDraw.append([round(x+130), round(y+200), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                            # color_array.append(COLORS[class_ids[i]])
                            count[1] = count[1] + 1
                            draw_bounding_box(img2_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes_covid,COLORS)


                        # net = cv2.dnn.readNetFromDarknet(cfg_person, weights_person)

                        # layer_names = net.getLayerNames()
                        # output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

                        blob = cv2.dnn.blobFromImage(img3, scale, (416,416), (0,0,0), True, crop=False)
                        # set input blob for the network
                        net.setInput(blob)

                        outs = net.forward(output_layers)
                        # print(outs)
                        


                        class_ids = []
                        confidences = []
                        boxes = []
                        conf_threshold = 0.5
                        nms_threshold = 0.7

                        

                        for out in outs:
                            for detection in out:
                                scores = detection[5:]
                                
                                class_idx = np.argmax(scores)
                                # print(classval)
                                confidence = scores[class_idx]
                                # print(confidence)
                                if confidence > 0.5:

                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)
                                    x = center_x - w / 2
                                    y = center_y - h / 2
                                    # class_idx = np.argmax(scores)
                                    if class_idx > (len(classes) - 1):
                                        class_idx = class_idx % (len(classes)+1)
                                    # print(classval)
                                    if class_idx == 0:
                                        class_ids.append(class_idx)
                                        confidences.append(float(confidence))
                                        boxes.append([x, y, w, h])


                        # boxes, confidences, conf_threshold, nms_threshold, class_ids = detect(img3, cfg, weights,classes)
                        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                    # print(class_ids)


                        classes_covid = ["violation","no_violation"]
                        # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                        path = "./Bounding_Boxes_Files/BB_prerecorded_S1"+str(ind_bb)+".txt"
                        f = open(path,'a')
                        # print(indices)
                        # print(boxes)
                        # print(ipm_matrix3)
                        circle_coord = []

                        # break
                        img3_detect = img3.copy()
                        box = [boxes[i] for i in indices]
                        # print(box)
                        # print("confidence::", confidences)
                        class_id_covid = []
                        # if len(box) == 1:


                        for i in range(len(box)-1):
                            if i + 1 != len(box):
                                dist = box[i][0] - box[i+1][0]
                                if dist > px_img2:
                                    class_id_covid.append(1)
                                else:
                                    class_id_covid.append(0)
                                x = box[i][0]
                                # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                                y =  box[i][1]
                                # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                                w = box[i][2] + box[i+1][2]
                                h = box[i][3]
                            else:
                                class_id_covid.append(1)

                            

                            # x = box[i][0]
                            # # # cv2.warpPerspective(box[0], ipm_matrix3, (1,1))[0,0]
                            # y =  box[i][1]
                            # # cv2.warpPerspective(box[1], ipm_matrix3, (1,1))[0,0]
                            # w = box[i][2]
                            # h = box[i][3]
                            # draw_bounding_box_base(img3, class_id_covid[i], confidences[i], round(x+50), round(y+180), round(x+w), round(y+h),classes_covid,COLORS)
                            toDraw.append([ round(x+50), round(y+180), round(x+w), round(y+h), class_id_covid[i], classes_covid, colors[class_id_covid[i]]])
                            # color_array.append(COLORS[class_ids[i]])
                            count[2] = count[2] + 1
                            draw_bounding_box(img3_detect, class_id_covid[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes_covid,COLORS)
                        


                        ipm_mat = []
                        ind_bb = ind_bb +1

                        # pts = np.array([[1080, 1000], [1080, 1920], [1, 1920], [1, 1]], dtype=np.float32)
                        # pts = np.array([[0,h],[w,h],[w,0],[0,0]],np.float32)
                        # pts = np.array([[493,133],[314,121],[274,130],[497,150]],np.float32)
                        # pts2 = np.array([[420,312],[398,178],[273,180],[278,311]],np.float32)
                        # pts3 = np.array([[84,214],[146,129],[84,120],[7,199]],np.float32)


                        pts = np.array([[494,477],[602,477],[537,452],[445,454]],np.float32)
                        pts = np.array([[793,799],[869,787],[915,783],[1037,779],[1101,775], [1205,773], [1263,771], [1361,769],[1709,761],[1421,707],[631,719],[797,652],[548,645]],np.float32)
                        pts2 = np.array([[27,1029],[1863,1019],[1209,832],[725,843]],np.float32)

                        pts2 = np.array([[397,782],[362,784],[320,791],[243,793],[206,797], [128,802], [72,804], [3,1078],[740,827]],np.float32)
                        # pts2 = np.array([[662,1023],[1049,1020],[1044,972],[743,969]],np.float32)
                        pts3 = np.array([[641,1070],[184,859],[823,801],[1297,867],[1369,998],[1155,908],[1077,825],[1024,818],[943,810]],np.float32)
                        # pts3 = np.array([[1255,1079],[1645,859],[1163,813],[781,810],[337,875], [1,949]],np.float32)
                        colors = [(0,0,255),(255,0,255),(0,255,255),(0,0,0),(134,134,255),(30,45,67)]
                        # i = 0
                        # for pt in pts3:
                        #     cv2.circle(img3, tuple(pt.astype(int)), 5, colors[i], -1)
                        #     i = i+1
                        #     if i == len(colors):
                        #         i = 0

                        ## compute IPM matrix and apply it
                        ## ipm_pts = np.array([[503,197],[650,197], [645,88], [510,88]], dtype=np.float32)

                        # ipm_pts = np.array([[521,321],[603,321], [612,270],[517,257]], dtype=np.float32)


                        # ipm_pts = np.array([[211,369],[311,369], [315,275],[211,281]], dtype=np.float32)
                        ipm_pts = np.array([[477,197],[472,197],[465,197],[461,197],[454,197],[449,197], [443,197], [440,198], [441,197],[410,248],[480,244],[412,453],[460,453]],np.float32)

                        # ipm_pts = np.array([[521,449],[603,451], [600,418],[521,418]], dtype=np.float32)
                        ipm_matrix,_ = cv2.findHomography(pts, ipm_pts,cv2.RANSAC,5.0)
                        # #ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                        ipm1 = cv2.warpPerspective(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1])
                        ipm_mat.append(ipm1)
                        # cv2.imwrite('img1_transformed.jpg',ipm1)



                        # ipm_pts2 = np.array([[519,246],[607,249], [609,202],[517,201]], dtype=np.float32)

                        # ipm_pts2 = np.array([[321,809],[327,291], [955,287],[975,791]], dtype=np.float32)
                        ipm_pts2 = np.array([[297,244],[474,245],[468,245],[460,245],[456,245], [450,245], [444,245], [413,275],[475,273]],np.float32)
                        ipm_matrix2,_ = cv2.findHomography(pts2, ipm_pts2,cv2.RANSAC,5.0)
                        # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                        ipm2 = cv2.warpPerspective(toRGBA(img2), ipm_matrix2, top_view.shape[:2][::-1])
                        ipm_mat.append(ipm2)
                      


                        # ipm_pts3 = np.array([[516,191],[611,185], [609,138],[520,140]], dtype=np.float32)


                        ipm_pts3 = np.array([[475,324],[408,321],[480,244], [410,248],[502,331],[473,330],[445,246],[439,246],[428,246]], dtype=np.float32)

                        # ipm_pts3 = np.array([[833,351],[985,783], [699,789],[423,795],[415,601],[409,405]], dtype=np.float32)
                        ipm_matrix3,_ = cv2.findHomography(pts3, ipm_pts3,cv2.RANSAC,5.0)
                        # ipm1 = cv2.estimateRigidTransform(toRGBA(img), ipm_matrix, top_view.shape[:2][::-1],fullAffine = True)
                        ipm3 = cv2.warpPerspective(toRGBA(img3), ipm_matrix3, top_view.shape[:2][::-1])
                        ipm_mat.append(ipm3)
                        # cv2.imwrite('img3_transformed.jpg',ipm3)


                        
                        # ipm_mat.append(ipm3)


                        toDrawPts = []
                        for i in range(0, count[0]):
                            #xpt = (toDraw[i][0] + toDraw[i][2])/2
                            #ypt = (toDraw[i][1] + toDraw[i][3])/2
                            #inter = np.matmul(ipm_matrix, np.array([round(xpt), round(ypt), 1]))
                            inter = np.matmul(ipm_matrix, np.array([toDraw[i][0], toDraw[i][1], 1]))
                            inter[0] = inter[0]/inter[2]
                            inter[1] = inter[1]/inter[2]
                            inter[2] = 1
                            toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                        for i in range(count[0], count[0]+count[1]):
                            #xpt = (toDraw[i][0] + toDraw[i][2])/2
                            #ypt = (toDraw[i][1] + toDraw[i][3])/2
                            #inter = np.matmul(ipm_matrix2, np.array([round(xpt), round(ypt), 1]))
                            inter = np.matmul(ipm_matrix2, np.array([toDraw[i][0], toDraw[i][1], 1]))
                            inter[0] = inter[0]/inter[2]
                            inter[1] = inter[1]/inter[2]
                            inter[2] = 1
                            toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])
                        for i in range(count[0]+count[1], count[0]+count[1]+count[2]):
                            #xpt = (toDraw[i][0] + toDraw[i][2])/2
                            #ypt = (toDraw[i][1] + toDraw[i][3])/2
                            #inter = np.matmul(ipm_matrix3, np.array([round(xpt), round(ypt), 1]))
                            inter = np.matmul(ipm_matrix3, np.array([toDraw[i][0], toDraw[i][1], 1]))
                            inter[0] = inter[0]/inter[2]
                            inter[1] = inter[1]/inter[2]
                            inter[2] = 1
                            toDrawPts.append([inter[0], inter[1], toDraw[i][4], toDraw[i][5], toDraw[i][6]])


                        #print("SHAPE", toDrawPts[0].shape, toDrawPts[0][0], toDrawPts[0][1], toDrawPts[0][2])




                        # print(ipm_mat)
                        images_array = [img,img2,img3]

                   




                        h,w,n = top_view.shape
                        top_view_sol = top_view.copy()
                        top_view_sol = toRGBA(top_view_sol)
                        top_view_sol_copy = top_view_sol.copy()
                        # cv2.imshow("Top View Actual", top_view)
                        # top_view_sol = np.hstack((top_view_sol,ipm_mat[2]))

                        # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                        # (status, stitched) = stitcher.stitch(images_array)

                        # cv2.imshow("stitched version", stitched)
                        # ipm_mat = ipm_mat[0]

                        # print(circle_coord[0][0])

                        for ipm in ipm_mat:
                            # ipm = toRGBA(ipm)
                            # print("Here")
                            for i in range(h):
                              for j in range(w):
                                location_px = ipm[i][j].copy()
                                if location_px[3] == 1:
                                     # and (top_view_sol[i][j]>0.7).all())
                                  #  and (location_px <0.1).any():
                                  top_view_sol[i][j] = location_px



                        for i in range(len(toDrawPts)):
                            draw_bounding_box_base_pt(top_view_sol, toDrawPts[i][2], toDrawPts[i][0], toDrawPts[i][1], toDrawPts[i][3], toDrawPts[i][4])

                        top_view = cv2.cvtColor(cv2.imread('Top_View.jpeg'),cv2.COLOR_BGR2RGB)/255
                                       # cv2.cvtColor(top_view_sol,cv2.COLOR_BGR2RGB)/255


                        # j = cv2.cvtColor(cv2.applyColorMap((gauss*255).astype(np.uint8),cv2.COLORMAP_SUMMER),cv2.COLOR_BGR2RGB).astype(np.float32)/255
                        k = 21
                            # x=0
                        gauss = cv2.getGaussianKernel(k,np.sqrt(64))
                        gauss = gauss * gauss.T
                        gauss = (gauss/gauss[int(k/2),int(k/2)])

                        heatmap = np.zeros((top_view.shape[0],top_view.shape[1],3)).astype(np.float32)
                        points = toDrawPts
                        # print(points)

                        for p in points:
                            # print(p[0]- int(k/2))
                            # print(p[0]+int(k/2)+1)
                            # print(p[1]- int(k/2))
                            # print(p[1]+int(k/2)+1)

                            # print(heatmap[int(p[0]):int(p[0]+int(k/2)+1),int(p[1]-int(k/2)):int(p[1]+int(k/2)+1),:].shape)
                            # print(j.shape)



                            try:
                                # b = heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:]
                                # c = j + b

                                # heatmap[int(p[0]):int(p[0]+int(k)),int(p[1]):int(p[1]+int(k)),:] = j
                                draw_bounding_box_base_pt(heatmap, p[2], p[0], p[1], p[3], p[4])
                            except:
                                pass

                        # m = np.max(heatmap, axis = (0,1))
                        # heatmap = heatmap/m




                        heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)

                        mask = np.where(heatmap > 0.2,1,0).astype(np.float32)
                        inv_mask = np.ones((top_view.shape[0],top_view.shape[1],3))*(1 - mask)[:,:,None]
                        
                        im = Image.fromarray((heatmap * 255).astype(np.uint8))
                        im.save('./frames_S1/frames'+str(ind)+".png")


                        images = cv2.imread('./frames_S1/frames'+str(ind)+".png", 0)
                        # colormap = plt.get_cmap('inferno')
                        # heatmap = (colormap(images) * 2**16).astype(np.uint16)[:,:,:3]
                        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

                        # heatmap = (heatmap/255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(images, cv2.COLORMAP_AUTUMN)
                        heatmap_array.append(heatmap)
                        # # imask = (inv_mask/255).astype(np.uint8)
                        # # imask = cv2.applyColorMap(imask, cv2.COLORMAP_HOT)
                        top_view = rgba2rgb(top_view)

                        
                        new_top = np.where(inv_mask>0,top_view,inv_mask)
                        new_top = inv_mask * top_view

                        try:
                            heatmap_new = heatmap_array[len(heatmap_array) - frames_to_consider]
                            for i in range(len(heatmap_array) - frames_to_consider+1,len(heatmap_array)):
                                heatmap_new = heatmap_new+heatmap_array[i]
                        except:
                            heatmap_new = heatmap_array[0]
                            for i in range(1,len(heatmap_array)):
                                heatmap_new = heatmap_new+heatmap_array[i]
                            # print("here")    


                        new_top = new_top + heatmap_new



                        # im = Image.fromarray((top_view_sol * 255).astype(np.uint8))
                        # im.save("./Prerecorded_Video_Heatmap_Covid/Top_View"+str(ind)+".png")
                        # im = cv2.imread(("./Prerecorded_Video_Heatmap_Covid/Top_View"+str(ind)+".png"))
                        # video.write(im)
                        ind = ind+1


                        cv2.imshow(windowName4,new_top)
                        cv2.imshow(windowName2,img3_detect)
                        cv2.imshow(windowName3,img2_detect)
                        cv2.imshow(windowName1,img_detect)

                        if cv2.waitKey(1) == 27:
                            break



                    # cv2.imshow("Actual img",img)
                    # im = Image.fromarray((heatmap * 255).astype(np.uint8))
                    # im.save('./frames_S1/frames'+str(ind)+".png")
                    # ind = ind + 1
            else:
                cv2.imshow(windowName1, frame1)
                cv2.imshow(windowName2, frame2)
                cv2.imshow(windowName3, frame3)

            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        capture3.release()
        capture2.release()
        cv2.destroyAllWindows()

    else:
        print("Invalid option entered. Exiting...")


main()
