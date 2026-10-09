#!/usr/bin/env python3
"""Author six held-out still-image tasks before native inference.

Pillow and a supplied DejaVu Sans font are generation-only dependencies.
Frozen images and their hashes are sufficient for runtime clients.
"""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if (args.output / 'tasks.json').exists():
        parser.error('preserve frozen tasks; use a new directory for regeneration')
    args.output.mkdir(parents=True, exist_ok=True)
    fonts = {size: ImageFont.truetype(str(args.font), size) for size in (28, 32, 36, 40, 48)}
    tasks = []

    def canvas(size):
        image = Image.new('RGB', size, 'white')
        return image, ImageDraw.Draw(image)

    def save(image, name, prompt, expected, category):
        path = args.output / name
        if path.exists():
            raise RuntimeError('preserve existing image: ' + name)
        if path.suffix == '.jpg':
            image.save(path, quality=95, subsampling=0)
        else:
            image.save(path)
        tasks.append({'name': path.stem, 'file': name, 'category': category,
            'held_out': True, 'size': list(image.size),
            'media_type': 'image/jpeg' if path.suffix == '.jpg' else 'image/png',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'prompt': prompt + ' Respond only with a JSON object containing these string fields: '
                + ', '.join(expected) + '. Do not add explanation or other fields.',
            'expected': expected})

    image, draw = canvas((640, 448))
    draw.text((135, 28), 'NOVA 582', font=fonts[48], fill='black')
    draw.ellipse((75, 160, 255, 340), fill=(255, 140, 0))
    draw.polygon([(445, 150), (340, 340), (550, 340)], fill=(128, 0, 128))
    save(image, 'nova-shapes.png', 'Read the heading exactly and identify the color and shape on each side.',
         {'heading': ['NOVA 582'], 'left': ['orange circle'], 'right': ['purple triangle']}, 'identifier/spatial')

    image, draw = canvas((640, 480))
    draw.rectangle((0, 0, 639, 479), fill=(232, 236, 242))
    draw.rounded_rectangle((40, 80, 600, 420), radius=14, fill='white', outline=(70, 80, 95), width=3)
    draw.rectangle((42, 82, 598, 130), fill=(42, 75, 120))
    draw.text((62, 90), 'Backup utility', font=fonts[28], fill='white')
    draw.text((72, 158), 'Backup completed', font=fonts[40], fill='black')
    draw.text((72, 232), 'Files saved: 18', font=fonts[36], fill='black')
    draw.rounded_rectangle((380, 322, 550, 388), radius=8, fill=(42, 75, 120))
    draw.text((416, 334), 'Close', font=fonts[32], fill='white')
    save(image, 'backup-dialog.png', 'What completion message, saved-file count, and button label are shown in the dialog?',
         {'title': ['Backup completed'], 'file_count': ['18'], 'button': ['Close']}, 'screenshot/dialog')

    image, draw = canvas((736, 512))
    draw.text((50, 28), 'Warehouse Stock', font=fonts[40], fill='black')
    xs, ys = [50, 460, 686], [112, 190, 270, 350, 430]
    draw.rectangle((50, 112, 686, 190), fill=(218, 230, 241))
    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill='black', width=3)
    for y in ys:
        draw.line((xs[0], y, xs[-1], y), fill='black', width=3)
    for y, item, units in [(130, 'Item', 'Units'), (210, 'Pencils', '14'), (290, 'Markers', '9'), (370, 'Notebooks', '23')]:
        draw.text((70, y), item, font=fonts[36], fill='black')
        draw.text((494, y), units, font=fonts[36], fill='black')
    save(image, 'warehouse-table.png', 'Read the table title. How many units of Markers are listed, and which item has the most units?',
         {'title': ['Warehouse Stock'], 'marker_units': ['9'], 'largest_item': ['Notebooks']}, 'table')

    image, draw = canvas((512, 768))
    draw.rectangle((18, 18, 493, 749), outline=(25, 100, 65), width=8)
    draw.text((76, 120), 'FIELD DAY', font=fonts[48], fill=(25, 100, 65))
    draw.text((62, 310), '14 MAY 2027', font=fonts[40], fill='black')
    draw.text((68, 460), 'CEDAR PARK', font=fonts[40], fill='black')
    save(image, 'field-day-portrait.png', 'Read the event name, date, and location from the poster.',
         {'event': ['FIELD DAY'], 'date': ['14 MAY 2027'], 'location': ['CEDAR PARK']}, 'portrait/poster')

    image, draw = canvas((641, 801))
    draw.text((48, 50), 'INVOICE', font=fonts[48], fill='black')
    for y, text in [(180, 'Invoice: AQ-417'), (290, 'Total: $86.50'), (400, 'Due: 09 JUN 2027')]:
        draw.text((48, y), text, font=fonts[36], fill='black')
    draw.line((48, 520, 588, 520), fill='black', width=2)
    draw.text((48, 570), 'Thank you', font=fonts[32], fill='black')
    save(image, 'invoice-unaligned.jpg', 'Read the invoice identifier, total amount, and due date.',
         {'invoice_id': ['AQ-417'], 'amount': ['$86.50', '86.50'],
          'due_date': ['09 JUN 2027', '9 JUN 2027', '09 JUNE 2027', '9 JUNE 2027']}, 'document/jpeg/resize')

    image, draw = canvas((640, 448))
    draw.text((155, 20), 'Shapes board', font=fonts[36], fill='black')
    for x in [80, 280, 480]:
        draw.polygon([(x + 40, 105), (x, 190), (x + 80, 190)], fill='black')
    draw.line((40, 230, 600, 230), fill=(160, 160, 160), width=2)
    for x in [170, 390]:
        draw.ellipse((x, 290, x + 80, 370), fill=(0, 80, 255))
    save(image, 'shape-counting.png', 'Count the triangles in the top row and circles in the bottom row. Identify the color of the bottom-row shapes.',
         {'top_triangles': ['3'], 'bottom_circles': ['2'], 'bottom_color': ['blue']}, 'counting/spatial')

    manifest = {'scope': 'six authored held-out image tasks frozen before native inference',
                'generation': {'pillow': __version__, 'font_sha256': hashlib.sha256(args.font.read_bytes()).hexdigest()},
                'tasks': tasks}
    (args.output / 'tasks.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('FROZEN_HELD_OUT_TASKS', len(tasks))


if __name__ == '__main__':
    main()
