# GSN_YOLO

## Autorzy:
  - Bartłomiej Moroz
  - Jan Walczak

## Struktura projektu:
- darknet.drawio - diagram naszego modelu Darknet53 w formacie draw.io
- darknet.py - definicja modelu Darknet53
- dataset.py - główny Dataset
- lightning_data.py - główny LightningDataModule
- lightning_model.py - głowny LightningModule
- loading_weights.py - wczytywanie wag z pliku
- main.py - punkt wejścia do programu
- nms.py - funkcje do post-processingu
- preprocess_once.py - skrypt do preprocessingu danych, używany do przetworzenia oryginalnych zipów na nasze
- primitive_dataloader.py - prosty, pomocniczy LightingDataModule
- primitive_dataset.py - prosty, pomocniczy Dataset
- README.md - tutaj jesteś
- requirements_colab.txt - plik z zależnościami projektu, wersja dla Google Colab (bez NumPy i Pillow, bo już jest w innej wersji)
- requirements.txt - plik z zależnościami projektu
- yolo_og_reference.ini - plik konfiguracyjny oryginalnego YOLOv3 (komentarze własne), używany tylko do referencji przy projektowaniu własnego modelu: źródło: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
- yolo_post.py - kolejne funkcje do post-processingu
- yolo.drawio - diagram naszego modelu YOLOv3 w formacie draw.io
- yolo.py - definicja modelu YOLOv3
