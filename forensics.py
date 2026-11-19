import cv2
import numpy as np
from sklearn.metrics import mutual_info_score
import sys


# This function scales a whole image or a single channel by alpha
def scale_image(image, alpha, center, pos):
    if alpha == 1:
        return image

    # cv2.imshow('roi', image)
    # cv2.waitKey(0)

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

    # cv2.imshow('crop_img', crop_img)
    # cv2.waitKey(0)

    ch1 = int(np.floor(ch * alpha))
    cw1 = int(np.floor(cw * alpha))
    img = cv2.resize(crop_img, (cw1, ch1))

    # cv2.imshow('scaled', img)
    # cv2.waitKey(0)

    if alpha < 1:
        top = (ch - ch1) // 2
        bot = ch - ch1 - top
        left = (cw - cw1) // 2
        right = cw - cw1 - left
        img = cv2.copyMakeBorder(img, top, bot, left, right,
                                 cv2.BORDER_CONSTANT, value=0)
        # cv2.imshow('scaled', img)
        # cv2.waitKey(0)

    if alpha > 1:
        top, left = 0, 0
        if px + w / 2 < cx:
            left = (cw1 - cw) // 2
        else:
            right = cw1 - cw - (cw1 - cw) // 2
            left = cw1 - w - right
        if py + h / 2 < cy:
            top = (ch1 - ch) // 2
        else:
            bot = ch1 - ch - (ch1 - ch) // 2
            top = ch1 - h - bot
        res = img[top:top + h, left:left + w]
        # cv2.imshow('result', res)
        # cv2.waitKey(0)
        return res
    else:
        res = img[p0y:p0y + h, p0x:p0x + w]
        # cv2.imshow('result', res)
        # cv2.waitKey(0)
        return res


def calc_MI_sum(image, pos, y, x, ab, ar):
    green = image[:, :, 1].flatten()
    ab_blue = scale_image(
        image[:, :, 0], 1 / ab, (y, x), pos).flatten()
    f = ab_blue > 0
    f_blue = ab_blue[f]
    f_green = green[f]
    mi_blue = 0
    if len(f_blue) > 0:
        hist = np.histogram2d(f_blue, f_green)[0]
        mi_blue = mutual_info_score(
            None, None, contingency=hist)

    ab_red = scale_image(
        image[:, :, 2], 1 / ar, (y, x), pos).flatten()
    f = ab_red > 0
    f_red = ab_red[f]
    f_green = green[f]
    mi_red = 0
    if len(f_red) > 0:
        hist = np.histogram2d(f_red, f_green)[0]
        mi_red = mutual_info_score(
            None, None, contingency=hist)

    return mi_blue + mi_red


def hill_climbing(image, pos, full_size, progress_cb=None):
    height, width = full_size[0], full_size[1]
    ab, ar = 1, 1
    y, x = height // 2, width // 2
    N_c = 3
    N = 3
    dxy = 25
    dalpha = 0.01

    def neighbours(y, x, ab, ar):
        n = []
        for i in range(N_c):
            for j in range(N_c):
                for k in range(N_c):
                    for l in range(N_c):
                        y_ = y - dxy * (N_c // 2) + i * dxy
                        x_ = x - dxy * (N_c // 2) + j * dxy
                        ab_ = ab - dalpha * (N_c // 2) + k * dalpha
                        ar_ = ar - dalpha * (N_c // 2) + l * dalpha
                        mi = calc_MI_sum(image, pos, y_, x_, ab_, ar_)
                        n.append((y_, x_, ab_, ar_, mi))
        return n

    for i in range(N):
        print('iteration {}/{}'.format(i + 1, N))
        mi_curr = calc_MI_sum(image, pos, y, x, ab, ar)
        while True:
            n = neighbours(y, x, ab, ar)
            best = max(n, key=lambda x: x[4])
            print('next best: {}'.format(best))
            if best[4] <= mi_curr:
                break

            y, x, ab, ar, mi_curr = best
        if progress_cb is not None:
            progress_cb((i + 1) / N)

        dxy //= 5
        dalpha /= 10
    return (y, x), ab, ar


# Find the best alpha
# pos = top-left point
def iterative(image, pos, full_size, progress_cb=None):
    height, width = full_size[0], full_size[1]
    # Best params: y, x, alpha_blue, alpha_red, MI
    best_p = (height // 2, width // 2, 1, 1, 0)

    N = 9
    min_y, max_y = 0, height - 1
    min_x, max_x = 0, width - 1
    incr_xy = 256
    alpha_min_blue, alpha_max_blue = 0.5, 1.5
    alpha_min_red, alpha_max_red = 0.5, 1.5
    incr_a = 0.1

    for i in range(N):
        print('iteration {}/{}'.format(i + 1, N))
        y = min_y
        y_total_iters = (max_y - min_y) / incr_xy
        y_iter = 0
        while y <= max_y:
            x = min_x
            while x <= max_x:
                ab = alpha_min_blue
                while ab <= alpha_max_blue:
                    ar = alpha_min_red
                    while ar <= alpha_max_red:
                        mi_sum = calc_MI_sum(image, pos, y, x, ab, ar)
                        # print('x {} y {}: {}'.format(x, y, mi_sum))
                        if mi_sum > best_p[4]:
                            best_p = (y, x, ab, ar, mi_sum)
                        ar += incr_a
                    ab += incr_a
                x += incr_xy
            y += incr_xy
            y_iter += 1
            if progress_cb is not None:
                progress_cb(i / N + (y_iter / y_total_iters) / N)

        print('best {}:{} alphas {}, {} mi {}'.format(
            best_p[0], best_p[1], best_p[2], best_p[3], best_p[4]))
        min_y = max(0, best_p[0] - incr_xy)
        max_y = min(height - 1, best_p[0] + incr_xy)
        min_x = max(0, best_p[1] - incr_xy)
        max_x = min(width - 1, best_p[1] + incr_xy)
        alpha_min_blue, alpha_max_blue = best_p[2] - incr_a, best_p[2] + incr_a
        alpha_min_red, alpha_max_red = best_p[3] - incr_a, best_p[3] + incr_a
        incr_xy //= 2
        incr_a /= 5

    center = (best_p[0], best_p[1])
    alpha_blue, alpha_red = best_p[2], best_p[3]
    # print('best mi %s' % best_p[4])

    # Correct blue and red channels
    # corr_blue = scale_image(image[:, :, 0], 1 / alpha_blue, center, pos)
    # corr_red = scale_image(image[:, :, 2], 1 / alpha_red, center, pos)

    # Create a new image from corrected blue, original green and corrected red
    # im_corr = cv2.merge((corr_blue, image[:, :, 1], corr_red))
    # cv2.imshow('corrected', im_corr)
    # cv2.waitKey(0)

    return (center, alpha_blue, alpha_red)


def draw_disp_arrows(image, center, alpha_blue, alpha_red):
    height, width = image.shape[0], image.shape[1]
    cy, cx = center
    image_blue_arrows = image.copy()
    slide_y, slide_x = int(height / 16), int(width / 16)
    scale_x, scale_y = 3, 3
    for i in range(16):
        for j in range(16):
            y = int(i * slide_y)
            x = int(j * slide_x)
            vx = (alpha_blue * (x - cx) + cx) - x
            vy = (alpha_blue * (y - cy) + cy) - y
            x1 = int(x + vx * scale_x)
            y1 = int(y + vy * scale_y)
            cv2.arrowedLine(image_blue_arrows,
                            (x, y), (x1, y1),
                            (255, 0, 0),
                            2)

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
    return image_blue_arrows, image_red_arrows


def find_forgeries(image, method, center, global_alpha_blue, global_alpha_red,
                   r_height, r_width, err_limit, show_rois, progress_cb=None):
    height, width = image.shape[0], image.shape[1]
    full_size = (height, width)
    slide_y, slide_x = r_height / 2, r_width / 2
    c0y, c0x = center
    rois = image.copy()
    max_error = 0
    rows = int(height / slide_y - 1)
    cols = int(width / slide_x - 1)

    for i in range(rows):
        for j in range(cols):
            start_x, start_y = int(j * slide_x), int(i * slide_y)
            end_x, end_y = int(start_x + r_width), int(start_y + r_height)
            roi = image[start_y:end_y, start_x:end_x]

            c1, a1, a2 = method(roi,
                                (start_y, start_x),
                                full_size)
            c2 = c1
            print('roi at {}:{} center {}:{} alphas {} {}'.format(
                i, j, c1[0], c1[1], a1, a2))

            c1y, c1x = c1
            c2y, c2x = c2

            # Calculate errors
            error = 0
            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    v0x = (global_alpha_blue * (x - c0x) + c0x) - x
                    v0y = (global_alpha_blue * (y - c0y) + c0y) - y
                    v1x = (a1 * (x - c1x) + c1x) - x
                    v1y = (a1 * (y - c1y) + c1y) - y
                    nom = np.dot([v0x, v0y], [v1x, v1y])
                    l0 = np.linalg.norm([v0x, v0y])
                    l1 = np.linalg.norm([v1x, v1y])
                    if l0 > 0.01 and l1 > 0.01:
                        denom = l0 * l1
                        e = (nom / denom)
                        e = max(min(1, e), -1)
                        error += np.arccos(e)

                    v0x = (global_alpha_red * (x - c0x) + c0x) - x
                    v0y = (global_alpha_red * (y - c0y) + c0y) - y
                    v1x = (a2 * (x - c2x) + c2x) - x
                    v1y = (a2 * (y - c2y) + c2y) - y
                    nom = np.dot([v0x, v0y], [v1x, v1y])
                    l0 = np.linalg.norm([v0x, v0y])
                    l1 = np.linalg.norm([v1x, v1y])
                    if l0 > 0.01 and l1 > 0.01:
                        denom = l0 * l1
                        e = (nom / denom)
                        e = max(min(1, e), -1)
                        error += np.arccos(e)

            error /= (r_width * r_height)
            if error > max_error:
                max_error = error
            print('error %s' % error)

            # Draw ROI rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (int(start_x),
                                      int(start_y + slide_y))
            fontScale = 0.6
            fontColor = (255, 255, 255)
            lineType = 1

            if error > err_limit:
                rect_color = (0, 0, 255)
                cv2.rectangle(rois, (start_x, start_y),
                              (end_x, end_y), rect_color, 2)
                cv2.putText(rois,
                            'error: {:.4f}'.format(error),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                if show_rois:
                    cv2.imshow('rois', rois)
            else:
                rect_color = (0, 255, 0)
                rois_ = rois.copy()
                cv2.rectangle(rois_, (start_x, start_y),
                              (end_x, end_y), rect_color, 2)
                cv2.putText(rois_,
                            'error: {:.4f}'.format(error),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                if show_rois:
                    cv2.imshow('rois', rois_)
            if show_rois:
                cv2.waitKey(0)
            if progress_cb is not None:
                progress_cb((i * cols + j + 1) / (rows * cols))
    print('max_error %s' % max_error)
    return rois, None


def usage():
    print('python main.py <filename> <method> [--show_rois]')
    print('arguments:')
    print('\tfilename: input filename')
    print('\tmethod: hill_climbing or iterative')
    print('\t--show_rois: display each ROI and wait for enter key')


if __name__ == '__main__':
    bad_input = len(sys.argv) != 4 and len(sys.argv) != 3
    method = None
    if not bad_input:
        method = sys.argv[2]
        bad_input = method != 'hill_climbing' and method != 'iterative'
        if len(sys.argv) == 4 and sys.argv[3] != '--show_rois':
            bad_input = True
    if bad_input:
        usage()
    else:
        method = hill_climbing if method == 'hill_climbing' else iterative
        show_rois = len(sys.argv) == 4
        image = cv2.imread(sys.argv[1])
        cv2.imshow('original', image)
        cv2.waitKey(0)
        center, alpha_blue, alpha_red = method(image, (0, 0), image.shape)

        # Correct blue and red channels
        corr_blue = scale_image(image[:, :, 0], 1 / alpha_blue, center, (0, 0))
        corr_red = scale_image(image[:, :, 2], 1 / alpha_red, center, (0, 0))

        # Create a new image from corrected blue, red, and original green
        im_corr = cv2.merge((corr_blue, image[:, :, 1], corr_red))
        cv2.imshow('corrected', im_corr)
        cv2.waitKey(0)

        print('global {}:{} blue: {} red: {}'.format(
            center[0], center[1], alpha_blue, alpha_red))
        b_arrows, r_arrows = draw_disp_arrows(
            image, center, alpha_blue, alpha_red)
        cv2.imshow('blue channel displacement', b_arrows)
        cv2.imshow('red channel displacement', r_arrows)
        cv2.waitKey(0)

        roi_height, roi_width = 200, 200
        find_forgeries(image, method, center, alpha_blue, alpha_red,
                       roi_height, roi_width, 5, show_rois)
        if not show_rois:
            cv2.waitKey(0)
