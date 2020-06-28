import cv2 as cv
import os
import sys
import utils
import yolo
import time

should_show_preview = False
should_write_result = True
detect_each_n_frame = 1
scale = 0.6

videoName = "springs_05"
videoExt = ".MOV"
# Net
sysPath = os.path.dirname(os.path.abspath(__file__))

weights = os.path.join(sysPath, "lib/yolov3-custom.weights")
config = os.path.join(sysPath, "lib/yolov3-custom.cfg")
names = os.path.join(sysPath, "lib/custom.names")

yo = yolo.Yolo(weights, config, names)
yo.confidence = 0.4
# Video
source = os.path.join(os.path.join(sysPath, "sources"), videoName + videoExt)
if not os.path.isfile(source):
    print("Input file ", source, " doesn't exist")
    sys.exit(1)

# source = 'http://devimages.apple.com/iphone/samples/bipbop/bipbopall.m3u8?dummy=param.mjpg'
# source = 'http://devimages.apple.com/iphone/samples/bipbop/gear1/prog_index.m3u8?dummy=param.mjpg'
# source = 'https://bitdash-a.akamaihd.net/content/sintel/hls/playlist.m3u8'
# source = 'https://bitmovin-a.akamaihd.net/content/playhouse-vr/m3u8s/105560.m3u8'
# source = 'https://bitdash-a.akamaihd.net/content/MI201109210084_1/m3u8s/f08e80da-bf1d-4e3d-8899-f0f6155f6efa.m3u8'
cap = cv.VideoCapture()
cap.open(source)

# check if we succeeded
if not cap.isOpened:
    sys.exit(1)

ret, img = cap.read()
height, width, _ = img.shape

capWrite = None
if should_write_result:
    resultFolder = os.path.join(sysPath, "result")
    if not utils.folderExist(resultFolder):
        os.mkdir(resultFolder)
    outSource = os.path.join(resultFolder, videoName + ".mp4")
    capWrite = cv.VideoWriter(
        outSource,
        0x7634706D,
        int(cap.get(cv.CAP_PROP_FPS)),
        (int(width * scale), int(height * scale)),
    )  # 0x7634706d - for mp4

i = 1
while True:
    ret, img = cap.read()
    if not ret:
        break
    if i % detect_each_n_frame == 0:  # each N frame, to fastener
        i = 0

        height, width, _ = img.shape
        im_res = cv.resize(img, (int(width * scale), int(height * scale)))
        img = im_res
        # show timing information on YOLO
        start = time.time()
        yo.detectFrom(img)
        end = time.time()
        print("[INFO] YOLO took {:.2f} seconds".format(end - start))

        # Visualize detected on source image
        font = cv.FONT_HERSHEY_PLAIN
        for obj in yo.objects:
            label = (
                obj.name
                + ": "
                + "{:.6f}".format(obj.confidence)
                + ": "
                + str(obj.x)
                + ", "
                + str(obj.y)
            )
            textColor = (0, 0, 0)
            boxColor = (150, 180, 20)
            cv.rectangle(
                img,
                (obj.x, obj.y),
                (obj.x + obj.width, obj.y + obj.height),
                boxColor,
                1,
            )
            cv.putText(img, label, (obj.x, obj.y - 5), font, 1, textColor, 2)

        if should_show_preview:
            cv.imshow("Video", img)

    if capWrite and should_write_result:
        capWrite.write(img)

    ch = cv.waitKey(1)
    if ch == 27:
        break
    i += 1


if capWrite:
    capWrite.release()
cap.release()
cv.destroyAllWindows()
