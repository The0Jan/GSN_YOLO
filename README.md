# GSN_YOLO

## Autorzy:
  - Bartłomiej Moroz
  - Jan Walczak

## Struktura projektu:
- darknet.drawio - diagram naszego modelu Darknet53 w formacie draw.io
- main.py - punkt wejścia do programu
- README.md - tutaj jesteś
- requirements_colab.txt - plik z zależnościami projektu, wersja dla Google Colab (bez NumPy i Pillow, bo już jest w innej wersji)
- requirements.txt - plik z zależnościami projektu
- src
  - datamodules
    - madai.py - główny LightningDataModule dla datasetu MADAI
    - primitive.py - prosty, pomocniczy LightingDataModule
  - datasets
    - główny Dataset MADAI
    - prosty, pomocniczy Dataset
  - lightning
    - głowny LightningModule
  - models
    - darknet.py - definicja modelu Darknet53
    - yolo.py - definicja modelu YOLOv3
  - other
    - loading_weights.py - wczytywanie wag z pliku
    - visualizing_results.py - wizualizacja predykcji modelu na zdjęciach
  - processing
    - anchor.py - funkcje do post-processingu surowego wyjścia z modelu, dopasowywanie anchorów, funkcja kosztu
    - nms.py - funkcje do redukcji liczby wyjściowych ramek, NMS
- tools
  - preprocess_dataset.py - skrypt do preprocessingu danych, używany do przetworzenia oryginalnych archiwów na nasze
- yolo.drawio - diagram naszego modelu YOLOv3 w formacie draw.io
- yolo_og_reference.ini - plik konfiguracyjny oryginalnego YOLOv3 (komentarze własne), używany tylko do referencji przy projektowaniu własnego modelu: źródło: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
