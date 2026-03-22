# Symplastic growth model (Python)

Обновлённая версия модели симпластического роста листа: перенос с Mathematica на Python с расширением функционала и **суррогатным моделированием** (ML/ИИ).

**Цель проекта — библиотека для моделирования:** исследователи (биологи, биophysicians) могут устанавливать пакет, задавать параметры, запускать симуляции и строить визуализации без правки исходного кода. API делается простым и стабильным, документация — по установке, параметрам и примерам. Подробнее: [docs/USAGE.md](docs/USAGE.md) (видение использования).

## Структура проекта

```
d:\ROOT\new\
├── symplastic_growth/     # ядро модели
│   ├── params.py         # параметры (GrowthParams)
│   ├── model.py          # ОДУ: осмотика, тургор, lr, segm
│   └── simulator.py       # SymplasticSimulator, run_until_length()
├── surrogate/             # суррогатное моделирование
│   ├── data.py           # генерация выборки (параметры → симулятор → X, y)
│   ├── surrogate.py      # GP и MLP-суррогаты
│   └── train.py          # обучение и оценка
├── notebooks/
│   ├── 01_symplastic_growth_demo.ipynb   # запуск модели и графики
│   └── 02_surrogate_training.ipynb      # обучение суррогата и сравнение
├── requirements.txt
└── README.md
```

## Установка

**Рекомендуется** — из корня репозитория, в виртуальном окружении:

```bash
cd d:\ROOT\new
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Так ставятся пакеты `symplastic_growth` и `surrogate` в режиме разработки, плюс `pytest`, Jupyter, PyYAML.

Классический вариант по `requirements.txt` (без editable install):

```bash
pip install -r requirements.txt
```

Только ядро без dev-зависимостей:

```bash
pip install -e .
# с визуализацией и Jupyter: pip install -e ".[viz,jupyter]"
# YAML-конфиги: pip install -e ".[config]"
```

### Тесты (трек A)

```bash
pytest -q
```

## Публикация и цитирование

- Лицензия: **MIT** (`LICENSE`).
- Как цитировать репозиторий: файл **`CITATION.cff`** (репозиторий: [github.com/zubairovaus/symplastic-growth](https://github.com/zubairovaus/symplastic-growth)).
- Пошаговый чеклист PyPI / Zenodo / JOSS: **[docs/TRACK_A.md](docs/TRACK_A.md)**.

## Быстрый старт

**Параметры из конфига (YAML/JSON):**

```python
from symplastic_growth import load_params, create_initial_leaf, run_until_length_multi

params = load_params("config_example.yaml")  # или свой my_config.yaml
leaf0 = create_initial_leaf(params)
res = run_until_length_multi(params, Ly=200.0, initial_leaf=leaf0)
```

В конфиге задают только нужные поля; остальные — по умолчанию. Пример: `config_example.yaml`. Для YAML: `pip install pyyaml`.

**Параметры в коде:**

```python
from symplastic_growth import GrowthParams, run_until_length

params = GrowthParams(alph=10.0, etha=0.15, thresh=2.0)
res = run_until_length(params, Ly=400.0, n_cells=8, dt=1.0)
print(res.total_length[-1], res.t[-1])
```

**Суррогат (обучить по выборке и предсказывать без запуска ОДУ):**

```python
from surrogate.data import generate_training_data
from surrogate.train import train_surrogate, evaluate_surrogate
from surrogate.surrogate import GPSurrogate

samples = generate_training_data(n_samples=50, Ly=300.0, n_cells=6, random_state=42)
gp = train_surrogate(samples, model_class=GPSurrogate)
# Дальше: gp.predict(X_new) или gp.predict_with_uncertainty(X_new)
```

## Суть модели (как в оригинале)

- **Осмотическое давление:** \( P_{osm} = \alpha (l_i - l)/l \).
- **Тургор (напряжение стенки):** \( P_{turg} \propto (l - l_r)/l_r \).
- **Рост расслабленной длины** \( l_r \): включается при \( P_{turg} > \text{thresh} \) и \( l_i > l \).
- **Сегменты:** длина сегмента меняется за счёт потока воды (разность осмотики и тургора по клеткам). Сегменты короче `min_fragment_length` (по умолчанию 0.1) сливаются с соседним сегментом той же клетки (JoinFragments1) — так модель ограничивает число очень мелких сегментов и сохраняет численную устойчивость при большом числе файлов клеток.

## Расширения

- Параметры собраны в **GrowthParams** — удобно менять и передавать в симулятор и в суррогат.
- **Суррогатное моделирование:** обучение по выборке запусков симулятора; предсказание выхода (длина, время) по параметрам без решения ОДУ. Реализованы:
  - **GP (Gaussian Process)** — даёт оценку неопределённости (std).
  - **MLP** — быстрый предсказатель без uncertainty.

## Мульти-файл и деление клеток

Полная симуляция с несколькими файлами и делением (по умолчанию включено):

```python
from symplastic_growth import GrowthParams, create_initial_leaf, run_until_length_multi, draw_leaf

params = GrowthParams()
params.n_cell_files = 2
params.smax = [10.0, 11.0]  # порог li для деления по файлам
leaf0 = create_initial_leaf(params, n_cells_per_file=[4, 4])
res = run_until_length_multi(params, Ly=100.0, initial_leaf=leaf0, with_division=True)
# res.final_leaf.total_cells растёт за счёт делений
draw_leaf(res.final_leaf, res.final_seg, params, mode='osm')
```

## План развития (научный проект)

Полный пошаговый план: **несколько файлов клеток**, **деление клеток**, **визуализация** и тесты — в **[docs/ROADMAP.md](docs/ROADMAP.md)**. Фазы 1–3 реализованы; далее: Фаза 4 (тесты), Фаза 5 (суррогат на полной модели).

## Jupyter

Откройте `notebooks/01_symplastic_growth_demo.ipynb` и `02_surrogate_training.ipynb` в Jupyter (из корня проекта `d:\ROOT\new`), выполните ячейки по порядку.
