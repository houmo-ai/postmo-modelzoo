import argparse
import json

import cv2


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img-path',
        dest='img_path',
        help='The source mage path',
    )
    parser.add_argument(
        '--output-path',
        dest='output_path',
        help='The path to store output image with boxes',
    )
    parser.add_argument(
        '--coco-names',
        dest='coco_names',
        help='The path coco names file',
    )
    parser.add_argument(
        '--detect-json',
        dest='detect_json',
        help='The detect json file path',
    )
    args = parser.parse_args()
    return args


def main(args=None):
    """main function"""
    if args is None:
        args = get_args()

    with open(args.detect_json) as det_file:
        detection = json.load(det_file)

    draw_bbox(args.output_path, args.img_path, detection, args.coco_names)


def draw_bbox(output_path, img_path, dets, coco_names_path):
    name_map = {}
    with open(coco_names_path) as names_file:
        for line_no, name in enumerate(names_file):
            name_map[line_no+1] = name.strip()
    color_map = {
        17: [0, 255, 255],
        2: [0, 0, 255],
        8: [0, 255, 0],
    }
    default_color = [127, 127, 255]
    rec_image = cv2.imread(img_path)
    for det in dets:
        img_id, l, t, w, h, prob, clazz = tuple(det)
        x1, y1, x2, y2 = int(l+0.5), int(t+0.5), int(l+w+0.5), int(t+h+0.5)
        rec_image = cv2.rectangle(
            rec_image, (x1, y1), (x2, y2), color_map.get(
                clazz, default_color,
            ), thickness=1, lineType=cv2.LINE_AA,
        )
        rec_image = cv2.putText(
            rec_image, name_map.get(
                clazz, 'Unknown',
            ), (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX,
            1, color_map.get(clazz, default_color), 2, cv2.LINE_AA,
        )
    cv2.imwrite(output_path, rec_image)


if __name__ == '__main__':
    main()
