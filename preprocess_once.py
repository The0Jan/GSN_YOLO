import xml.etree.ElementTree as ET
import os
import shutil

classLUT = {
    'aircraft': '0',
    'aircraftw': '0',               # typo in the dataset!!!
    'bomber': '1',
    'bomberw': '1',                 # typo in the dataset!!!
    'bomberww': '1',                # typo in the dataset!!!
    'Tu95': '1',                    # typo in the dataset!!!
    'early_warning_aircraft': '2',
    'fighter': '3',
    'fighterw': '3',                # typo in the dataset!!!
    'mili_helicopter': '4',
    'Helicopter': '4',              # typo in the dataset!!!
    'holicopter': '4',              # typo in the dataset!!!
}

def process_xml(filename):
    root = ET.parse(filename).getroot()
    size = root.find('size')
    boxes = [[size.find('width').text, size.find('height').text]]
    for child in root.findall('object'):
        name = child.find('name')
        bndbox = child.find('bndbox')
        boxes.append([classLUT[name.text], bndbox.find('xmin').text, bndbox.find('ymin').text, bndbox.find('xmax').text, bndbox.find('ymax').text])
    return boxes

def process_annotations(filename, outdir):
    boxes = process_xml(filename)
    name = os.path.basename(filename)
    name, _ = os.path.splitext(name)
    with open(os.path.join(outdir, name + '.csv'), 'w') as out:
        out.writelines([','.join(box) + '\n' for box in boxes])

if __name__ == "__main__":
    shutil.unpack_archive('train-MADAI.zip', 'train-MADAI', format='zip')
    shutil.unpack_archive('test-MADAI.zip', 'test-MADAI', format='zip')

    inpath = 'train-MADAI/Train_Annotations'
    outpath = 'train/annotations'
    os.makedirs(outpath, exist_ok=True),
    for file in os.listdir(inpath):
        process_annotations(os.path.join(inpath, file), outpath)

    inpath = 'test-MADAI/Annotations'
    outpath = 'test/annotations'
    os.makedirs(outpath, exist_ok=True),
    for dir in os.listdir(inpath):
        for file in os.listdir(os.path.join(inpath, dir)):
            process_annotations(os.path.join(inpath, dir, file), outpath)

    shutil.make_archive('train-new', format='zip', root_dir='train', base_dir='.')
    shutil.make_archive('test-new', format='zip', root_dir='test', base_dir='.')
