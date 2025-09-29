# -*- coding: utf-8 -*-
"""
HRSC2016 XML → DOTA labelTxt（一次产出 L1/L2/L3 三套）
- 严格匹配 HRSC 标注结构: HRSC_Image/HRSC_Objects/HRSC_Object
- 读取字段: Class_ID, mbox_cx, mbox_cy, mbox_w, mbox_h, mbox_ang, difficult
- L3: 依据 hrsc-mmrotate.py 中 HRSC_CLASSES/HRSC_CLASSES_ID
- L2: 将 L3 归并到四大类（aircraft carrier / warship / merchant ship / submarine）
- L1: 固定为 ship
- 输出 DOTA 行: x1 y1 x2 y2 x3 y3 x4 y4 <category> <difficult>

用法：
    python hrsc_to_dota_all_levels.py \
        --root /path/to/HRSC2016/Train \
        --out_prefix labelTxt \
        --strict_l2   # 可选：L2无法归并时，跳过该目标（默认回退为 ship）
    # 对 Test 分割同理

如需把 L3 类名做训练友好化（去空格/特殊字符），加：
    --sanitize_l3
"""

import os
import os.path as osp
import argparse
import re
import xml.etree.ElementTree as ET

# 你工程里已有：从旋转框五元组(cx,cy,w,h,ang)转四点多边形
# dota_poly2rbox.py
import math

def rbox2poly_single(rbox):
    """
    rbox: (cx, cy, w, h, ang_rad)  # ang 为弧度
    return: [x1, y1, x2, y2, x3, y3, x4, y4]
    顶点顺序：左上 -> 右上 -> 右下 -> 左下（按未旋转时的顺序旋转后返回）
    """
    cx, cy, w, h, ang = rbox
    dx, dy = w / 2.0, h / 2.0
    pts = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    ca, sa = math.cos(ang), math.sin(ang)
    poly = []
    for x, y in pts:
        xr = x * ca - y * sa + cx
        yr = x * sa + y * ca + cy
        poly.extend([xr, yr])
    return poly



# ========= 取自 hrsc-mmrotate.py：类别与两位ID一一对应 =========
HRSC_CLASSES = (
    'ship', 'aircraft carrier', 'warcraft', 'merchant ship',
    'Nimitz', 'Enterprise', 'Arleigh Burke', 'WhidbeyIsland',
    'Perry', 'Sanantonio', 'Ticonderoga', 'Kitty Hawk',
    'Kuznetsov', 'Abukuma', 'Austen', 'Tarawa', 'Blue Ridge',
    'Container', 'OXo|--)', 'Car carrier([]==[])',
    'Hovercraft', 'yacht', 'CntShip(_|.--.--|_]=', 'Cruise',
    'submarine', 'lute', 'Medical', 'Car carrier(======|',
    'Ford-class', 'Midway-class', 'Invincible-class'
)

HRSC_CLASSES_ID = (
    '01','02','03','04','05','06','07','08','09','10','11','12','13','14',
    '15','16','17','18','19','20','22','24','25','26','27','28','29',
    '30','31','32','33'
)
# 注意：这里的两位ID与类名按顺序对应；XML中的 Class_ID 形如 '1000000' + 两位ID


def build_id_maps():
    """构建：
      - full_id -> L3_name  (e.g., '100000013' -> 'Kuznetsov')
      - two_digit_id -> L3_name (e.g., '13' -> 'Kuznetsov')
    """
    two2name = {sid: name for sid, name in zip(HRSC_CLASSES_ID, HRSC_CLASSES)}
    full2name = {'1000000' + sid: name for sid, name in two2name.items()}
    return full2name, two2name


def l3_to_l2(name: str) -> str:
    """将 HRSC L3 名称归并成 L2 四类（小写字符串）。"""
    n = name.strip().lower()

    # 直接判别
    if 'submarine' in n:
        return 'submarine'
    if 'aircraft carrier' in n or 'carrier' in n:
        return 'aircraft carrier'
    if n == 'warcraft':
        return 'warship'
    if n == 'merchant ship':
        return 'merchant ship'

    # 型号关键词归并
    carrier_keys = [
        'nimitz', 'enterprise', 'kitty hawk', 'kuznetsov',
        'ford-class', 'midway-class', 'invincible-class'
    ]
    if any(k in n for k in carrier_keys):
        return 'aircraft carrier'

    warship_keys = [
        'arleigh burke', 'whidbeyisland', 'perry', 'sanantonio',
        'ticonderoga', 'abukuma', 'austen', 'tarawa', 'blue ridge'
    ]
    if any(k in n for k in warship_keys):
        return 'warship'

    merchant_keys = [
        'container', 'cntship', 'car carrier', 'cruise', 'yacht',
        'hovercraft', 'medical', 'lute', 'oxo|--)'  # 特殊记号类归为民船
    ]
    if any(k in n for k in merchant_keys):
        return 'merchant ship'

    # 其他未知：回退为 ship（可在严格模式下改为 None）
    if n == 'ship':
        return 'ship'
    return 'ship'


_SANITIZE_RE = re.compile(r'[^0-9a-zA-Z_]+')

def sanitize_label(s: str) -> str:
    """
    训练友好化：转小写，空格→下划线，去掉奇异符号。
    例如：'Car carrier([]==[])' -> 'car_carrier'
    """
    s = s.strip().lower().replace(' ', '_')
    s = _SANITIZE_RE.sub('_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'ship'


def parse_xml_items(xml_path: str):
    """解析单个 XML，返回每个目标的 (poly8, class_id_str, difficult)。"""
    items = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[WARN] parse failed: {xml_path} ({e})")
        return items

    parent = root.find('HRSC_Objects')
    if parent is None:
        return items

    for obj in parent.findall('HRSC_Object'):
        class_id = (obj.findtext('Class_ID') or '').strip()

        diff = 1 if (obj.findtext('difficult') or '0').strip() == '1' else 0
        try:
            cx = float((obj.findtext('mbox_cx') or '').strip())
            cy = float((obj.findtext('mbox_cy') or '').strip())
            w  = float((obj.findtext('mbox_w')  or '').strip())
            h  = float((obj.findtext('mbox_h')  or '').strip())
            ang= float((obj.findtext('mbox_ang')or '').strip())
        except Exception:
            # 缺字段：跳过该目标
            continue

        poly = rbox2poly_single((cx, cy, w, h, ang))  # [x1,y1,...,x4,y4]
        items.append((poly, class_id, diff))
    return items


def write_labeltxt(path: str, rows):
    """rows: list of 'x1 y1 ... x4 y4 label difficult'"""
    if not rows:
        open(path, 'w', encoding='utf-8').close()
        return
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')


def convert_split(root_dir: str,
                  out_prefix: str = 'labelTxt',
                  strict_l2: bool = False,
                  sanitize_l3: bool = False):
    """
    在 root_dir 下（如 /HRSC2016/Train），输出：
      - {out_prefix}_L1/
      - {out_prefix}_L2/
      - {out_prefix}_L3/
    """
    img_dir = osp.join(root_dir, 'images')
    ann_dir = osp.join(root_dir, 'Annotations')

    out_l1 = osp.join(root_dir, f'{out_prefix}_L1')
    out_l2 = osp.join(root_dir, f'{out_prefix}_L2')
    out_l3 = osp.join(root_dir, f'{out_prefix}_L3')
    os.makedirs(out_l1, exist_ok=True)
    os.makedirs(out_l2, exist_ok=True)
    os.makedirs(out_l3, exist_ok=True)

    full2l3, _ = build_id_maps()

    img_stems = [osp.splitext(n)[0] for n in os.listdir(img_dir)
                 if n.lower().endswith(('.bmp', '.jpg', '.png', '.tif', '.tiff'))]

    for stem in img_stems:
        xml_path = osp.join(ann_dir, f'{stem}.xml')

        # 三个输出文件
        p_l1 = osp.join(out_l1, f'{stem}.txt')
        p_l2 = osp.join(out_l2, f'{stem}.txt')
        p_l3 = osp.join(out_l3, f'{stem}.txt')

        if not osp.exists(xml_path):
            # 无标注：写空文件，保持对齐
            write_labeltxt(p_l1, [])
            write_labeltxt(p_l2, [])
            write_labeltxt(p_l3, [])
            continue

        items = parse_xml_items(xml_path)

        rows_l1, rows_l2, rows_l3 = [], [], []

        for poly, class_id, diff in items:
            x1, y1, x2, y2, x3, y3, x4, y4 = poly

            # L3 名称
            l3_name = full2l3.get(class_id, 'ship')
            l3_for_write = sanitize_label(l3_name) if sanitize_l3 else l3_name

            # L2 名称
            l2_name = l3_to_l2(l3_name)
            if l2_name == 'ship' and strict_l2:
                # 严格模式：无法归并的（如 L3=ship）直接跳过
                pass
            else:
                rows_l2.append(f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {l2_name} {diff}")

            # L1 固定
            rows_l1.append(f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} ship {diff}")
            # L3 原始
            rows_l3.append(f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {l3_for_write} {diff}")

        write_labeltxt(p_l1, rows_l1)
        write_labeltxt(p_l2, rows_l2)
        write_labeltxt(p_l3, rows_l3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='HRSC2016 split root, e.g., /path/HRSC2016/Train or /Test')
    ap.add_argument('--out_prefix', default='labelTxt', help='output dir prefix: {prefix}_L1/_L2/_L3')
    ap.add_argument('--strict_l2', action='store_true',
                    help='if set, skip objects that cannot be merged to L2 (default: fallback to "ship")')
    ap.add_argument('--sanitize_l3', action='store_true',
                    help='if set, normalize L3 names to training-friendly tokens (lower/underscore/no symbols)')
    args = ap.parse_args()

    convert_split(args.root, out_prefix=args.out_prefix,
                  strict_l2=args.strict_l2, sanitize_l3=args.sanitize_l3)
    print('done.')


if __name__ == '__main__':
    main()
