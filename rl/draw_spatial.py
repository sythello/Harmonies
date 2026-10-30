"""Render the architecture diagram (Pillow; available in the desktop document runtime)."""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
image = Image.new('RGB', (1400, 1260), '#f6f8fb')
draw = ImageDraw.Draw(image)
font_dir = Path('/System/Library/Fonts/Supplemental')


def text(x, y, label, size=25, bold=False, color='#163047', anchor='mm'):
    font = ImageFont.truetype(str(font_dir / ('Arial Bold.ttf' if bold else 'Arial.ttf')), size)
    draw.text((x, y), label, font=font, fill=color, anchor=anchor)


def box(x, y, w, h, title, subtitle='', fill='#ffffff'):
    draw.rounded_rectangle((x, y, x+w, y+h), 16, fill=fill, outline='#c6d1dd', width=2)
    text(x+w/2, y+h/2-(16 if subtitle else 0), title, 25, True)
    if subtitle: text(x+w/2, y+h/2+20, subtitle, 21)


def arrow(points, color='#71869a'):
    draw.line(points, fill=color, width=3, joint='curve')
    (x0, y0), (x, y) = points[-2:]
    angle = math.atan2(y-y0, x-x0)
    draw.polygon([(x, y), (x-12*math.cos(angle-.45), y-12*math.sin(angle-.45)),
                  (x-12*math.cos(angle+.45), y-12*math.sin(angle+.45))], fill=color)


text(50, 50, 'v003  |  Spatial FFN', 38, True, anchor='lm')
text(50, 95, '2,900,467 parameters  /  shared board filters  /  four independent target towers', 23, anchor='lm')
box(50, 145, 300, 95, 'Numeric input: 1614', 'Original features retained')
box(430, 145, 540, 95, 'Extract both boards', '2 players x 23 cells x 29 categories', '#eaf4f4')
box(1050, 145, 300, 95, 'Market: 5 card IDs')
arrow([(350, 190), (430, 190)])
arrow([(700, 240), (700, 290)])
draw.rounded_rectangle((430, 290, 970, 510), 16, fill='#eaf4f4', outline='#c6d1dd', width=2)
text(700, 323, '5 kernels x 16 channels + ReLU', 25, True)
import json
schema = json.loads((ROOT/'rl/spatial.json').read_text())
for i, kernel in enumerate(schema['kernels']):
    coordinates = [(q*17, (r+q*.5)*19.63) for q, r in kernel['offsets']]
    cx, cy = sum(x for x, y in coordinates)/len(coordinates), sum(y for x, y in coordinates)/len(coordinates)
    for x, y in coordinates:
        x, y = 490 + i*105 + x-cx, 395+y-cy
        polygon = [(x+10*math.cos(math.pi*j/3), y+10*math.sin(math.pi*j/3)) for j in range(6)]
        draw.polygon(polygon, fill='#42978e', outline='#eaf4f4', width=2)
    text(490+i*105, 455, ['3 / A', '3 / B', '6 / A', '6 / B', '7 / hex'][i], 19)
text(700, 487, 'Boundary: separate category #30', 19)
arrow([(700, 510), (700, 540)])
box(430, 540, 540, 90, 'Flatten: 3680', '5 kernels x 2 players x 23 anchors x 16', '#eaf4f4')
arrow([(1200, 240), (1200, 300)])
box(1050, 300, 300, 95, 'Shared embedding', '5 x 8 = 40')
arrow([(200, 240), (200, 730), (425, 730)])
text(215, 635, '1614', 22, anchor='lm')
arrow([(1200, 395), (1200, 730), (975, 730)])
text(1185, 635, '40', 22, anchor='rm')
arrow([(700, 630), (700, 690)])
box(425, 690, 550, 80, 'Concat: 5334', '1614 + 40 + 3680', '#e7edf8')
arrow([(700, 770), (700, 825)])
box(425, 825, 550, 70, 'Linear 5334 -> 512 + ReLU', fill='#e7edf8')
arrow([(700, 895), (700, 930)])
box(425, 930, 550, 70, 'Linear 512 -> 128 + ReLU', fill='#e7edf8')
for x, title, output in [(50, 'Policy', '885 logits'), (390, 'Rank', '2 values + tanh'),
                         (730, 'Relative score', '2 linear values'), (1070, 'Absolute score', '2 linear values')]:
    arrow([(700, 1000), (700, 1035), (x+140, 1035), (x+140, 1080)])
    box(x, 1080, 280, 100, title, '128 -> 64 + ReLU', '#fff1de')
    text(x+140, 1210, '64 -> '+output, 20)
target = ROOT/'docs/diagrams/rl-spatial-v3.png'
target.parent.mkdir(parents=True, exist_ok=True)
image.save(target)
print(target)
