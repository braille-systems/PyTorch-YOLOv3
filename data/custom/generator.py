"""
Applies transformations on tiles.
Download set of images, use them as background for tiles to train object detection.
"""

import itertools
import os
import random
import zipfile
from dataclasses import dataclass
from typing import List, Generator, Iterable, Dict

import wget
from PIL import Image

Img = Image.Image
TRANSPARENT = (255, 255, 255, 0)


@dataclass
class Label:
    x: int  # Top left corner
    y: int  # Top right corner
    w: int
    h: int
    c: int  # Number of class


@dataclass
class Tile:
    img: Img
    label: Label


@dataclass
class DataImage:
    img: Img
    labels: List[Label]


GenTiles = Generator[Tile, None, None]


def download_backgrounds(path: str, link: str) -> None:
    if os.path.isdir(path):
        print('Needed data has been already downloaded!')
        return

    os.makedirs(path)
    zip_path = os.path.join(path, 'data.zip')
    wget.download(link, zip_path)
    print('Downloaded! Unzipping...')
    with zipfile.ZipFile(os.path.join(zip_path), 'r') as zf:
        zf.extractall(path)
    os.remove(zip_path)


def get_name_to_i_class(path: str) -> Dict[str, int]:
    res = {}
    with open(path) as file:
        for i, line in enumerate(file):
            res[line.strip()] = i
    return res


def read_tiles(folder: str, name_to_i_class: Dict[str, int]) -> GenTiles:
    while True:
        for name in (f for f in os.listdir(folder)
                     if os.path.isfile(os.path.join(folder, f))):
            image = Image.open(os.path.join(folder, name))
            w, h = image.size
            class_name, *_ = name.split('.')
            yield Tile(image, Label(x=0, y=0, w=w, h=h, c=name_to_i_class[class_name]))


def apply_rotations(tiles: Iterable[Tile], angles: List[int]) -> GenTiles:
    for tile, angle in zip(tiles, itertools.cycle(angles)):
        yield from apply_rotations_helper(tile, angle)


def apply_rotations_helper(tile: Tile, angle: int) -> GenTiles:
    for a in [angle, -angle]:
        rotated = tile.img.rotate(a, expand=True)
        w, h = rotated.size
        yield Tile(rotated, Label(x=0, y=0, w=w, h=h, c=tile.label.c))


def apply_resize(tiles: Iterable[Tile], sizes: List[int]) -> GenTiles:
    for tile, size in zip(tiles, itertools.cycle(sizes)):
        yield from apply_resize_helper(tile, size)


def apply_resize_helper(tile: Tile, size: int) -> GenTiles:
    old_size = max(tile.img.size)
    square = Image.new(tile.img.mode, (old_size, old_size), color=TRANSPARENT)
    square.paste(tile.img, (0, 0))

    label = Label(
        x=0, y=0,
        w=int(size / old_size * tile.label.w),
        h=int(size / old_size * tile.label.h),
        c=tile.label.c
    )

    yield Tile(square.resize((size, size)), label)


def apply_perspective(tiles: Iterable[Tile]) -> GenTiles:
    yield from tiles  # TODO


def place_fragments_on_bg(
        tiles: Iterable[Tile], bg_path: str, *,
        n_fragments_lo: int, n_fragments_hi: int, img_size: int
) -> Generator[DataImage, None, None]:
    for filename in (f for f in os.listdir(bg_path)
                     if os.path.isfile(os.path.join(bg_path, f))):
        n_fragments = random.randrange(n_fragments_lo, n_fragments_hi)
        background = Image.open(os.path.join(bg_path, filename))
        crop_size = min(background.size)
        square_background = Image.new(background.mode, (crop_size, crop_size))
        square_background.paste(background, (0, 0))
        yield from place_fragments_on_bg_helper(
            take(tiles, n_fragments),
            square_background.resize((img_size, img_size))
        )


def place_fragments_on_bg_helper(
        tiles: Iterable[Tile], background: Img, n_tries: int = 5
) -> Generator[DataImage, None, None]:
    labels = []
    for tile in tiles:
        for _ in range(n_tries):
            w, h = background.size
            x = random.randrange(w - tile.label.w)
            y = random.randrange(h - tile.label.h)
            label = Label(x=x, y=y, w=tile.label.w, h=tile.label.h, c=tile.label.c)
            if not is_intersect(label, labels):
                labels.append(label)
                background.paste(tile.img, (x, y), tile.img)
                break

    yield DataImage(
        background,
        labels
    )


def take(xs: Iterable, n) -> Generator:
    for _, x in zip(range(n), xs):
        yield x


def is_in_rect(x: int, y: int, label: Label) -> bool:
    return label.x <= x <= label.x + label.w and label.y <= y <= label.y + label.h


def is_intersect_helper(a: Label, b: Label) -> bool:
    return any([
        is_in_rect(a.x, a.y, b),
        is_in_rect(a.x + a.w, a.y, b),
        is_in_rect(a.x, a.y + a.h, b),
        is_in_rect(a.x + a.w, a.y + a.h, b),
        is_in_rect(a.x + a.w // 3, a.y + a.h // 2, b),
        is_in_rect(a.x + 2 * a.w // 3, a.y + a.h // 2, b),
        is_in_rect(a.x + a.w // 2, a.y + a.h // 3, b),
        is_in_rect(a.x + a.w // 2, a.y + 2 * a.h // 3, b),
    ])


def is_intersect(x: Label, ys: List[Label]) -> bool:
    for y in ys:
        if is_intersect_helper(x, y):
            return True
    return False


def write(data_images: Iterable[DataImage], img_path: str, lbl_path: str, n_images: int) -> None:
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(lbl_path, exist_ok=True)
    i = 0
    while i < n_images:
        for data_image in data_images:
            print(f'Image: {i} of {n_images}')
            data_image.img.save(os.path.join(img_path, f'{i}.png'))
            with open(os.path.join(lbl_path, f'{i}.txt'), 'w') as f:
                for label in data_image.labels:
                    size, _ = data_image.img.size
                    assert size == _
                    cx = label.x + label.w / 2
                    cy = label.y + label.h / 2
                    f.write(f'{label.c} {cx / size} {cy / size} {label.w / size} {label.h / size}' + '\n')

            i += 1


def write_annotations(folder: str, train_path: str, valid_path: str, n_train: int, n_valid: int) -> None:
    with open(train_path, 'w') as f:
        for i in range(n_train):
            f.write(os.path.join(folder, f'{i}.png') + '\n')

    with open(valid_path, 'w') as f:
        for i in range(n_train, n_train + n_valid):
            f.write(os.path.join(folder, f'{i}.png') + '\n')


def main() -> None:
    size = 500
    n_images = 5000
    fragment_sizes = [100, 120, 150, 180, 190, 200]
    angles = [5, 6, 7, 8, 10, 15, 16, 17, 20, 25, 30]
    bg_url = 'http://images.cocodataset.org/zips/val2017.zip'
    n_fragments_lo = 3
    n_fragments_hi = 15
    valid = 0.2

    tiles_folder = os.path.join('raw')
    bg_folder = os.path.join('background', 'val2017')
    res_images_path = os.path.join('images')
    res_labels_path = os.path.join('labels')
    classes_path = os.path.join('classes.names')

    download_backgrounds(bg_folder, bg_url)

    name_to_i_class = get_name_to_i_class(classes_path)
    tiles = read_tiles(tiles_folder, name_to_i_class)
    tiles = apply_rotations(tiles, angles)
    tiles = apply_resize(tiles, fragment_sizes)
    tiles = apply_perspective(tiles)

    data_images = place_fragments_on_bg(
        tiles, bg_folder,
        n_fragments_lo=n_fragments_lo,
        n_fragments_hi=n_fragments_hi,
        img_size=size)
    write(data_images, res_images_path, res_labels_path, n_images)

    write_annotations(
        folder='data/custom/images',
        train_path='train.txt',
        valid_path='valid.txt',
        n_train=int((1 - valid) * n_images),
        n_valid=int(valid * n_images)
    )


if __name__ == '__main__':
    main()
