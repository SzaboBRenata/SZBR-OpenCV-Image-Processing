import cv2
import numpy as np
from sklearn.metrics import mutual_info_score
 
 
# This function scales a whole image or a single channel by alpha
def scale_image(image, alpha, center, pos):
    if alpha == 1:
        return image
 
    cy, cx = center
    py, px = pos
    h = image.shape[0]
    w = image.shape[1]
    ch = 2 * max(abs(cy - py), abs(cy - (py + h)))
    cw = 2 * max(abs(cx - px), abs(cx - (px + w)))
 
    crop_img = np.zeros((ch, cw), np.uint8)
    p0x, p0y = 0, 0
    if px + w / 2 <= cx:
        p0x = 0
    else:
        p0x = cw - w
    if py + h / 2 <= cy:
        p0y = 0
    else:
        p0y = ch - h
    crop_img[p0y:p0y + h, p0x:p0x + w] = image[:, :]
 
    ch1 = int(np.floor(ch * alpha))
    cw1 = int(np.floor(cw * alpha))
    img = cv2.resize(crop_img, (cw1, ch1))
 
    if alpha < 1:
        top = (ch - ch1) // 2
        bot = ch - ch1 - top
        left = (cw - cw1) // 2
        right = cw - cw1 - left
        img = cv2.copyMakeBorder(img, top, bot, left, right,
                                 cv2.BORDER_CONSTANT, value=0)
 
    if alpha > 1:
        top, left = 0, 0
        if px < cx:
            left = (cw1 - cw) // 2
        else:
            right = cw1 - cw - (cw1 - cw) // 2
            left = cw1 - w - right
        if py < cy:
            top = (ch1 - ch) // 2
        else:
            bot = ch1 - ch - (ch1 - ch) // 2
            top = ch1 - h - bot
        res = img[top:top + h, left:left + w]
        return res
    else:
        res = img[p0y:p0y + h, p0x:p0x + w]
        return res
 
 
# Find the best alpha
# pos = top-left point
def find_params(image, center, pos):  # kép, kp, kezdőpont
    green = image[:, :, 1].flatten()
 
    incr = 0.01
    N = 1
    alpha_min_blue, alpha_max_blue = 0.5, 1.5
    alpha_min_red, alpha_max_red = 0.5, 1.5
    alpha_blue, alpha_red = 1, 1
 
    for i in range(N):
        print('%s iteration' % i)
        imCA = []
        alpha = alpha_min_blue
        while alpha <= alpha_max_blue:
            ab_blue = scale_image(
                image[:, :, 0], 1 / alpha, center, pos).flatten()
 
            # with mutual information
            hist_blue = np.histogram2d(ab_blue, green)[0]
            mi_blue = mutual_info_score(None, None, contingency=hist_blue)
            imCA.append((alpha, mi_blue))
 
            alpha += incr
 
        # Find the alpha values with best score
        alpha_blue = max(imCA, key=lambda x: x[1])[0]
 
        imCA = []
        alpha = alpha_min_red
        while alpha <= alpha_max_red:
            ab_red = scale_image(
                image[:, :, 2], 1 / alpha, center, pos).flatten()
 
            hist_red = np.histogram2d(ab_red, green)[0]
            mi_red = mutual_info_score(None, None, contingency=hist_red)
            imCA.append((alpha, mi_red))
 
            alpha += incr
 
        alpha_red = max(imCA, key=lambda x: x[1])[0]
 
        print('best blue %s best red %s' % (alpha_blue, alpha_red))
 
        alpha_min_blue, alpha_max_blue = alpha_blue - incr, alpha_blue + incr
        alpha_min_red, alpha_max_red = alpha_red - incr, alpha_red + incr
        incr /= 10
 
    # Correct blue and red channels
    # corr_blue = scale_image(image[:, :, 0], 1 / alpha_blue, center, pos)
    # corr_red = scale_image(image[:, :, 2], 1 / alpha_red, center, pos)
 
    # Create a new image from corrected blue, original green and corrected red
    # im_corr = cv2.merge((corr_blue, image[:, :, 1], corr_red))
    # cv2.imshow('corrected', im_corr)
    # cv2.waitKey(0)
 
    return (alpha_blue, alpha_red)
 
 
def draw_disp_arrows(image, center, alpha_blue, alpha_red): #megkapja a képet...
    height, width = image.shape[0], image.shape[1]
    cy, cx = center
    image_blue_arrows = image.copy()
    slide_y, slide_x = int(height / 16), int(width / 16)  #16 nyilam lesz mindkét irányba
    scale_x, scale_y = 3, 3 #3 a nyíl nagysaga
    for i in range(16):
        for j in range(16):
            y = int(i * slide_y)
            x = int(j * slide_x)
            vx = (alpha_blue * (x - cx) + cx) - x #vektor értéke v0 v1
            vy = (alpha_blue * (y - cy) + cy) - y
            x1 = int(x + vx * scale_x)#pont ahol vagyok - vektor koordináták a rajzoláskoz
            y1 = int(y + vy * scale_y)
            cv2.arrowedLine(image_blue_arrows,
                            (x, y), (x1, y1),
                            (255, 0, 0),
                            2)
    cv2.imshow('blue channel displacement', image_blue_arrows) #kék nyilak rajzolása
 
    image_red_arrows = image.copy()
    slide_y, slide_x = int(height / 16), int(width / 16)
    for i in range(16):
        for j in range(16):
            y = int(i * slide_y)
            x = int(j * slide_x)
            vx = (alpha_red * (x - cx) + cx) - x
            vy = (alpha_red * (y - cy) + cy) - y
            x1 = int(x + vx * scale_x)
            y1 = int(y + vy * scale_y)
            cv2.arrowedLine(image_red_arrows,
                            (x, y), (x1, y1),
                            (0, 0, 255),
                            2)
    cv2.imshow('red channel displacement', image_red_arrows) #piros nyíl rajz
    cv2.waitKey(0)
 
 
def find_forgeries(image, center, global_alpha_blue, global_alpha_red):#kis téglalapok
    height, width = image.shape[0], image.shape[1]
    r_height, r_width = height / 4, width / 4
    slide_y, slide_x = r_height / 2, r_width / 2 #mennyivel mozgatom jobbra vagy le a négyzetet
    cy, cx = center
 
    for y in range(0, int(height / slide_y - 1)): #lefele ennyiszer mozgatom
        for x in range(0, int(width / slide_x - 1)):
            rois = image.copy() 
            start_x, start_y = int(x * slide_x), int(y * slide_y)
            end_x, end_y = int(start_x + r_width), int(start_y + r_height)
            cv2.rectangle(rois, (start_x, start_y),
                          (end_x, end_y), (255, 0, 0), 2) #bgr  2 a vonal vastagsaga
            roi = image[start_y:end_y, start_x:end_x]
            # cv2.imshow('roi', roi)
            # cv2.waitKey(0)
            alpha_blue, alpha_red = find_params(roi, center, #center az egész kép közepe
                                                (start_y, start_x)) # mindig a bal felső sarok
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (int(start_x),
                                      int(start_y + slide_y // 2))
            fontScale = 0.6
            fontColor = (255, 255, 255)
            lineType = 2
            cv2.putText(rois,
                        'blue: {:.4f} red: {:.4f}'.format(
                            alpha_blue, alpha_red),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.imshow('rois', rois)
            cv2.waitKey(0)
 
            # Calculate errors
            # error = 0
            # for y1 in range(start_y, end_y):
            #     for x1 in range(start_x, end_x):
            #         v0x = (global_alpha_red * (x1 - cx) + cx) - x1
            #         v0y = (global_alpha_red * (y1 - cy) + cy) - y1
            #         v1x = (alpha_red * (x1 - cx) + cx) - x1
            #         v1y = (alpha_red * (y1 - cy) + cy) - y1
            #         nom = np.dot([v0x, v0y], [v1x, v1y])
            #         denom = np.linalg.norm(
            #             [v0x, v0y]) * np.linalg.norm([v1x, v1y])
            #         e = (nom / denom)
            #         e = min(max(1, e), -1)
            #         error += np.arccos(e)
            # print('error %s' % error)
 
 
image = cv2.imread('chromatic.jpg')   #itt indul
cv2.imshow('original', image)
cv2.waitKey(0)
alpha_blue, alpha_red = find_params(image,
                                    (image.shape[0] // 2,   #kép közepe
                                     image.shape[1] // 2),
                                    (0, 0)) # bal felső sarok koordinátája
print('global blue: {} red: {}'.format(alpha_blue, alpha_red))
draw_disp_arrows(image,
                 (image.shape[0] // 2, image.shape[1] // 2),
                 alpha_blue,
                 alpha_red)
find_forgeries(image,
               (image.shape[0] // 2, image.shape[1] // 2),
               alpha_blue,
               alpha_red)

