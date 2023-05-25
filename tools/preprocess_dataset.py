"""
Nazwa: preprocess_dataset.py
Opis: Skrypt do jednorazowego przetwarzania wstępnego zbioru danych
      na format, który nam bardziej odpowiada. Adnotacje z XML do CSV,
      zmiana struktury folderów.
Autor: Bartłomiej Moroz, Jan Walczak
"""
import os
from shutil import make_archive, unpack_archive, copyfile
from typing import List
from xml.etree.ElementTree import parse as xml_parse, Element


classLUT = {
    "aircraft": "0",
    "aircraftw": "0",  # typo in the dataset!!!
    "bomber": "1",
    "bomberw": "1",  # typo in the dataset!!!
    "bomberww": "1",  # typo in the dataset!!!
    "Tu95": "1",  # typo in the dataset!!!
    "early_warning_aircraft": "2",
    "fighter": "3",
    "fighterw": "3",  # typo in the dataset!!!
    "mili_helicopter": "4",
    "Helicopter": "4",  # typo in the dataset!!!
    "holicopter": "4",  # typo in the dataset!!!
}


def process_xml(filename: str) -> List[List[str]]:
    """
    Extract meaningful data (bounding boxes and class) from XML file to array.
    """
    root = xml_parse(filename).getroot()
    boxes = []
    for child in root.findall("object"):
        name = child.find("name")
        if not isinstance(name, Element) or name.text is None:
            raise RuntimeError(f"XML File {filename} is missing <name> or its content!")
        bndbox = child.find("bndbox")
        if not isinstance(bndbox, Element):
            raise RuntimeError(f"XML File {filename} is missing <bndbox>!")
        xmin, xmax = bndbox.find("xmin"), bndbox.find("xmax")
        ymin, ymax = bndbox.find("ymin"), bndbox.find("ymax")
        if any(
            [
                not isinstance(x, Element) or x.text is None
                for x in (xmin, xmax, ymin, ymax)
            ]
        ):
            raise RuntimeError(f"XML File {filename} is missing <bndbox>!")
        boxes.append([classLUT[name.text], xmin.text, ymin.text, xmax.text, ymax.text])  # type: ignore
    return boxes


def process_annotations(filename: str, outdir: str) -> None:
    """
    Dump annotation array to CSV file.
    """
    boxes = process_xml(filename)
    name = os.path.basename(filename)
    name, _ = os.path.splitext(name)
    with open(os.path.join(outdir, name + ".csv"), "w") as out:
        out.writelines([",".join(box) + "\n" for box in boxes])


def process_images(filename: str, outdir: str) -> None:
    """
    Just copy images to output directory.
    """
    copyfile(filename, os.path.join(outdir, os.path.basename(filename)))


if __name__ == "__main__":
    TRAIN_ZIP_IN = "train-MADAI"
    TRAIN_ZIP_OUT = "train-new"
    TRAIN_CSV_IN = os.path.join(TRAIN_ZIP_IN, "Train_Annotations")
    TRAIN_CSV_OUT = os.path.join(TRAIN_ZIP_OUT, "annotations")
    TRAIN_IMG_IN = os.path.join(TRAIN_ZIP_IN, "Train_JPEGImages")
    TRAIN_IMG_OUT = os.path.join(TRAIN_ZIP_OUT, "images")

    TEST_ZIP_IN = "test-MADAI"
    TEST_ZIP_OUT = "test-new"
    TEST_CSV_IN = os.path.join(TEST_ZIP_IN, "Annotations")
    TEST_CSV_OUT = os.path.join(TEST_ZIP_OUT, "annotations")
    TEST_IMG_IN = os.path.join(TEST_ZIP_IN, ".")
    TEST_IMG_OUT = os.path.join(TEST_ZIP_OUT, "images")

    unpack_archive(TRAIN_ZIP_IN + ".zip", TRAIN_ZIP_IN, format="zip")
    os.makedirs(TRAIN_CSV_OUT, exist_ok=True)
    os.makedirs(TRAIN_IMG_OUT, exist_ok=True)
    for file in os.listdir(TRAIN_CSV_IN):
        process_annotations(os.path.join(TRAIN_CSV_IN, file), TRAIN_CSV_OUT)
    for file in os.listdir(TRAIN_IMG_IN):
        process_images(os.path.join(TRAIN_IMG_IN, file), TRAIN_IMG_OUT)
    make_archive(TRAIN_ZIP_OUT, format="zip", root_dir=TRAIN_ZIP_OUT, base_dir=".")

    unpack_archive(TEST_ZIP_IN + ".zip", TEST_ZIP_IN, format="zip")
    os.makedirs(TEST_CSV_OUT, exist_ok=True)
    os.makedirs(TEST_IMG_OUT, exist_ok=True)
    for dir in os.listdir(TEST_CSV_IN):
        for file in os.listdir(os.path.join(TEST_CSV_IN, dir)):
            process_annotations(os.path.join(TEST_CSV_IN, dir, file), TEST_CSV_OUT)
    for dir in os.listdir(TEST_IMG_IN):
        # Do not process 'Annotations' dir
        if dir == os.path.basename(TEST_CSV_IN):
            continue
        for file in os.listdir(os.path.join(TEST_IMG_IN, dir)):
            process_images(os.path.join(TEST_IMG_IN, dir, file), TEST_IMG_OUT)
    make_archive(TEST_ZIP_OUT, format="zip", root_dir=TEST_ZIP_OUT, base_dir=".")
