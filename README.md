<div align="center">

# 🎯 Detection Library

**Мини-библиотека для локализации и детекции объектов с упором на YOLOv3**

**A mini library for object localization & detection, built around YOLOv3**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv3](https://img.shields.io/badge/Model-YOLOv3-00A67E?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Pascal%20VOC-1F6FEB?style=for-the-badge)

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/sdfgsgsjghjfgh/yolov3)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Weights-FFD21E?style=flat-square)](https://huggingface.co/danil-ml-2026/YOLOv3-VOC/tree/main)

🇷🇺 [Русский](#-русская-версия) · 🇬🇧 [English](#-english-version)

</div>

---

# 🇷🇺 Русская версия

## 📖 О проекте

Небольшая библиотека для задач **локализации и детекции объектов**. Основной упор сделан на **YOLOv3**: в репозитории есть сама модель, функция потерь, датасет и загрузчик для Pascal VOC, аугментации, NMS и подсчёт метрики mAP.

## ✨ Возможности

- 🧠 **YOLOv3**: полная модель, лёгкий backbone `Darknet53S` и полноценный классификатор `Darknet53` для предобучения
- 📉 **Функции потерь**: `YOLOv3Loss` и другие loss-функции
- 📊 **Метрики**: подсчёт mAP на валидации через `ap_per_class`
- 🗂️ **Данные**: обработка и кеширование Pascal VOC, готовый `Dataset` для PyTorch
- 🎨 **Аугментации** для загрузчика данных
- ⚡ **Векторизованный NMS** для YOLOv3
- 📐 **Утилиты для боксов**: IoU (попарный и «каждый с каждым»), конвертация форматов и масштабов

## 🗂️ Структура проекта

```text
Education_localization
├── loss
│   └── loss.py             # YOLOv3Loss и другие loss-функции
├── models
│   └── yolov3.py           # YOLOv3, Darknet53S, Darknet53
├── metrics
│   └── metrics.py          # ap_per_class (mAP на валидации)
└── utils
    ├── augmentation.py     # аугментации для loader
    ├── bbox_iou.py         # box_iou, bbox_iou
    ├── convert_box.py      # конвертация форматов и масштабов боксов
    ├── dataloaders.py      # PascalVOC, DatasetPascalVocYOLOv3
    ├── general.py          # grid_xy
    └── nms.py              # nms_yolov3_vectorized
```

## 🧩 Описание модулей

### 📉 `loss/loss.py`

| Объект | Описание |
|---|---|
| `YOLOv3Loss` | Основная функция потерь для YOLOv3 |
| другие loss-функции | Вспомогательные функции потерь |

### 🧠 `models/yolov3.py`

| Класс | Описание |
|---|---|
| `YOLOv3` | Вся модель детекции целиком |
| `Darknet53S` | Backbone для YOLOv3 |
| `Darknet53` | Полноценный классификатор, используется для предобучения |

### 📊 `metrics/metrics.py`

| Функция | Описание |
|---|---|
| `ap_per_class` | Подсчёт mAP на валидации |

### 🛠️ `utils/`

| Файл | Что внутри |
|---|---|
| `augmentation.py` | Аугментации для загрузчика данных |
| `bbox_iou.py` | `box_iou`: сравнение каждого бокса с каждым; `bbox_iou`: попарное сравнение |
| `convert_box.py` | Функции для конвертации между форматами и масштабами боксов |
| `dataloaders.py` | `PascalVOC`: первичная обработка и кеширование в удобный формат; `DatasetPascalVocYOLOv3`: приведение к формату `Dataset` из torch |
| `general.py` | `grid_xy`: распределение таргетов по сетке в формате YOLOv3 |
| `nms.py` | `nms_yolov3_vectorized`: векторизованная версия алгоритма NMS |

## 🚀 Быстрый старт

```bash
git clone https://github.com/fffffopas/Detection_library.git
cd Detection_library
```

Импорты (запускать из корня репозитория):

```python
from Education_localization.models.yolov3 import YOLOv3, Darknet53S, Darknet53
from Education_localization.loss.loss import YOLOv3Loss
from Education_localization.metrics.metrics import ap_per_class
from Education_localization.utils.dataloaders import PascalVOC, DatasetPascalVocYOLOv3
from Education_localization.utils.nms import nms_yolov3_vectorized
```

## 📓 Обучение и проверка

Полный код обучения и валидации находится в ноутбуке на Kaggle:

👉 **[YOLOv3 на Kaggle](https://www.kaggle.com/code/sdfgsgsjghjfgh/yolov3)**

## 🏋️ Веса модели

Обученные веса доступны на Hugging Face:

👉 **[danil-ml-2026/YOLOv3-VOC](https://huggingface.co/danil-ml-2026/YOLOv3-VOC/tree/main)**

<!-- Сюда можно добавить таблицу с результатами (mAP@0.5 и т.д.), когда будут финальные цифры -->

---

# 🇬🇧 English version

## 📖 About

A small library for **object localization and detection** tasks, with a strong focus on **YOLOv3**. The repository includes the model itself, the loss function, a Pascal VOC dataset and loader, augmentations, NMS, and mAP evaluation.

## ✨ Features

- 🧠 **YOLOv3**: the full model, a lightweight `Darknet53S` backbone, and a full `Darknet53` classifier for pretraining
- 📉 **Loss functions**: `YOLOv3Loss` and other loss functions
- 📊 **Metrics**: validation mAP via `ap_per_class`
- 🗂️ **Data**: Pascal VOC preprocessing and caching, plus a ready-to-use PyTorch `Dataset`
- 🎨 **Augmentations** for the data loader
- ⚡ **Vectorized NMS** for YOLOv3
- 📐 **Box utilities**: IoU (pairwise and all-vs-all), format and scale conversions

## 🗂️ Project structure

```text
Education_localization
├── loss
│   └── loss.py             # YOLOv3Loss and other loss functions
├── models
│   └── yolov3.py           # YOLOv3, Darknet53S, Darknet53
├── metrics
│   └── metrics.py          # ap_per_class (validation mAP)
└── utils
    ├── augmentation.py     # loader augmentations
    ├── bbox_iou.py         # box_iou, bbox_iou
    ├── convert_box.py      # box format and scale conversions
    ├── dataloaders.py      # PascalVOC, DatasetPascalVocYOLOv3
    ├── general.py          # grid_xy
    └── nms.py              # nms_yolov3_vectorized
```

## 🧩 Modules

### 📉 `loss/loss.py`

| Object | Description |
|---|---|
| `YOLOv3Loss` | The main YOLOv3 loss |
| other loss functions | Auxiliary loss functions |

### 🧠 `models/yolov3.py`

| Class | Description |
|---|---|
| `YOLOv3` | The complete detection model |
| `Darknet53S` | Backbone for YOLOv3 |
| `Darknet53` | Full classifier used for pretraining |

### 📊 `metrics/metrics.py`

| Function | Description |
|---|---|
| `ap_per_class` | Computes mAP on the validation set |

### 🛠️ `utils/`

| File | Contents |
|---|---|
| `augmentation.py` | Augmentations for the data loader |
| `bbox_iou.py` | `box_iou`: compares every box with every other box; `bbox_iou`: pairwise comparison |
| `convert_box.py` | Functions for converting between box formats and scales |
| `dataloaders.py` | `PascalVOC`: initial processing and caching into a convenient format; `DatasetPascalVocYOLOv3`: adapts the data to the torch `Dataset` interface |
| `general.py` | `grid_xy`: assigns targets to the YOLOv3 grid |
| `nms.py` | `nms_yolov3_vectorized`: vectorized NMS implementation |

## 🚀 Quick start

```bash
git clone https://github.com/fffffopas/Detection_library.git
cd Detection_library
```

Imports (run from the repository root):

```python
from Education_localization.models.yolov3 import YOLOv3, Darknet53S, Darknet53
from Education_localization.loss.loss import YOLOv3Loss
from Education_localization.metrics.metrics import ap_per_class
from Education_localization.utils.dataloaders import PascalVOC, DatasetPascalVocYOLOv3
from Education_localization.utils.nms import nms_yolov3_vectorized
```

## 📓 Training & evaluation

The full training and validation code lives in a Kaggle notebook:

👉 **[YOLOv3 on Kaggle](https://www.kaggle.com/code/sdfgsgsjghjfgh/yolov3)**

## 🏋️ Model weights

Trained weights are available on Hugging Face:

👉 **[danil-ml-2026/YOLOv3-VOC](https://huggingface.co/danil-ml-2026/YOLOv3-VOC/tree/main)**

<!-- Add a results table (mAP@0.5, etc.) here once you have final numbers -->

---

<div align="center">

⬆️ [Наверх / Back to top](#-detection-library)

</div>
