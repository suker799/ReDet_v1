# -*- coding: utf-8 -*-
"""
DOTA(labelTxt) → COCO for HRSC2016 (L1/L2/L3)
- 目录结构要求（与 HRSC2COCO.py 相同）：
  srcpath/
    images/    <-- *.bmp（或你切片后的图片）
    labelTxt/  <-- DOTA文本（x1 y1 ... x4 y4 cls difficult）

- level: l1 / l2 / l3
- sanitized: L3 类名是否做过 sanitize（与 hrsc_to_dota_all_levels.py 的 --sanitize_l3 对应）
- 若没有 dota_utils，可用内置兜底解析器

Example:
  python dota_to_coco_multilevel.py \
    --srcpath /root/project/redet/HRSC_DOTA_L2/train \
    --destfile /root/project/redet/HRSC_DOTA_L2/train.json \
    --level l2

  python dota_to_coco_multilevel.py \
    --srcpath /root/project/redet/HRSC_DOTA_L3/train \
    --destfile /root/project/redet/HRSC_DOTA_L3/train.json \
    --level l3 --sanitized

参考：你上传的 HRSC2COCO.py。 
"""

import os
import os.path as osp
import json
from PIL import Image

# ========= 1) 类别定义 =========

L1_NAMES = ['ship']

# L2 四类（与我们之前输出 labelTxt_L2 完全一致）
L2_NAMES = ['aircraft carrier', 'warship', 'merchant ship', 'submarine']

# L3 原始类名（与 hrsc-mmrotate.py 一致，含空格和特殊符号）
L3_NAMES_RAW = [
    'ship', 'aircraft carrier', 'warcraft', 'merchant ship',
    'Nimitz', 'Enterprise', 'Arleigh Burke', 'WhidbeyIsland',
    'Perry', 'Sanantonio', 'Ticonderoga', 'Kitty Hawk',
    'Kuznetsov', 'Abukuma', 'Austen', 'Tarawa', 'Blue Ridge',
    'Container', 'OXo|--)', 'Car carrier([]==[])',
    'Hovercraft', 'yacht', 'CntShip(_|.--.--|_]=', 'Cruise',
    'submarine', 'lute', 'Medical', 'Car carrier(======|',
    'Ford-class', 'Midway-class', 'Invincible-class'
]

# 若你在生成 L3 的 DOTA 时用了 --sanitize_l3，则类名会变成更训练友好的写法
L3_NAMES_SANITIZED = [
    'ship','aircraft_carrier','warcraft','merchant_ship','nimitz','enterprise',
    'arleigh_burke','whidbeyisland','perry','sanantonio','ticonderoga',
    'kitty_hawk','kuznetsov','abukuma','austen','tarawa','blue_ridge',
    'container','oxo','car_carrier','hovercraft','yacht','cntship','cruise',
    'submarine','lute','medical','car_carrier_eq','ford_class','midway_class','invincible_class'
]


# ========= 2) dota_utils 兼容层（若缺失则使用兜底实现） =========

try:
    import dota_utils as util  # 与 HRSC2COCO.py 一致
except Exception:
    # --- 兜底实现：足以解析 x1 y1 x2 y2 x3 y3 x4 y4 name difficult 的 DOTA 行 ---
    import glob
    import math

    class _Util:
        @staticmethod
        def GetFileFromThisRootDir(root_dir, ext='.txt'):
            files = glob.glob(osp.join(root_dir, f'*{ext}'))
            return sorted(files)

        @staticmethod
        def custombasename(path):
            return osp.splitext(osp.basename(path))[0]

        @staticmethod
        def _poly_area(poly_xy):
            # poly_xy: [x1,y1,...,x4,y4]
            xs = poly_xy[0::2]
            ys = poly_xy[1::2]
            s = 0.0
            for i in range(len(xs)):
                j = (i + 1) % len(xs)
                s += xs[i] * ys[j] - xs[j] * ys[i]
            return abs(s) / 2.0

        @staticmethod
        def parse_dota_poly2(label_file):
            """
            返回 list of dict:
               {'name': str, 'poly':[x1,y1,..,x4,y4], 'area': float}
            支持多空格/换行；忽略空行
            """
            objects = []
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 10:
                        # 至少 8 坐标 + 类别 + difficult
                        continue
                    try:
                        coords = list(map(float, parts[:8]))
                    except Exception:
                        continue
                    name = parts[8]
                    # difficult = parts[9]  # 暂不使用
                    area = _Util._poly_area(coords)
                    objects.append({'name': name, 'poly': coords, 'area': area})
            return objects

    util = _Util()


# ========= 3) 主逻辑：DOTA → COCO =========

def build_categories(class_names):
    cats = []
    for idx, name in enumerate(class_names):
        cats.append({'id': idx + 1, 'name': name, 'supercategory': name})
    return cats


def build_info(desc='HRSC (DOTA→COCO)'):
    return {
        'contributor': 'converted by dota_to_coco_multilevel.py',
        'description': desc,
        'version': '1.0',
        'year': 2025
    }


def convert_one_split(srcpath, destfile, class_names, strict=False):
    """
    srcpath: 包含 images/ 与 labelTxt/ 的目录
    class_names: 用于 categories 与 name→category_id 的查找
    strict: 若 True，遇到 labelTxt 中的 name 不在 class_names 里则跳过该实例
    """
    image_dir = osp.join(srcpath, 'images')
    label_dir = osp.join(srcpath, 'labelTxt')

    data = {
        'info': build_info(),
        'images': [],
        'categories': build_categories(class_names),
        'annotations': []
    }

    image_id = 1
    ann_id = 1

    # 遍历 labelTxt/*.txt
    label_files = util.GetFileFromThisRootDir(label_dir)
    for lf in label_files:
        stem = util.custombasename(lf)
        # 支持 bmp/jpg/png（优先 bmp）
        img_path = None
        for ext in ('.bmp', '.jpg', '.png', '.tif', '.tiff', '.jpeg'):
            p = osp.join(image_dir, stem + ext)
            if osp.exists(p):
                img_path = p
                break
        if img_path is None:
            # 没找到图片就跳过
            continue

        # 读尺寸
        with Image.open(img_path) as img:
            width, height = img.width, img.height

        data['images'].append({
            'file_name': osp.basename(img_path),
            'id': image_id,
            'width': width,
            'height': height
        })

        # 解析 DOTA 对象
        objects = util.parse_dota_poly2(lf)
        for obj in objects:
            name = obj['name']
            if name not in class_names:
                if strict:
                    continue
                else:
                    # 若非 strict，尝试宽松匹配（比如大小写差异）
                    # 也可以在此处添加自定义 name 标准化
                    try_names = [name, name.lower(), name.replace('_', ' ')]
                    matched = None
                    for t in try_names:
                        if t in class_names:
                            matched = t
                            break
                    if matched is None:
                        continue
                    name = matched

            cat_id = class_names.index(name) + 1
            poly = obj['poly']
            xmin, ymin = min(poly[0::2]), min(poly[1::2])
            xmax, ymax = max(poly[0::2]), max(poly[1::2])
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

            data['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'segmentation': [poly],
                'area': float(obj['area']),
                'bbox': bbox,
                'iscrowd': 0
            })
            ann_id += 1

        image_id += 1

    # 写出 COCO JSON
    os.makedirs(osp.dirname(destfile), exist_ok=True)
    with open(destfile, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f"[OK] saved COCO to {destfile}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--srcpath', required=True, help='directory containing images/ and labelTxt/')
    ap.add_argument('--destfile', required=True, help='output COCO json path')
    ap.add_argument('--level', choices=['l1', 'l2', 'l3'], required=True, help='which level to convert')
    ap.add_argument('--sanitized', action='store_true',
                    help='set if your L3 labelTxt used sanitized class names')
    ap.add_argument('--strict', action='store_true',
                    help='drop any instance whose class not in class_names')
    args = ap.parse_args()

    if args.level == 'l1':
        classes = L1_NAMES
    elif args.level == 'l2':
        classes = L2_NAMES
    else:
        classes = L3_NAMES_SANITIZED if args.sanitized else L3_NAMES_RAW

    convert_one_split(args.srcpath, args.destfile, classes, strict=args.strict)


if __name__ == '__main__':
    main()
