# B12 Cluster Workflow in ORCA 6.1

Многостадийный расчетный конвейер для поиска низкоэнергетических структур нейтрального кластера `B12` с использованием схемы:

`multi-start geometry generation -> GFN2-xTB screening -> unique minima filtering -> r2SCAN-3c refinement -> NumFreq validation`

Репозиторий предназначен для воспроизводимого запуска на Linux/VPS и хранения всех промежуточных и итоговых артефактов: стартовых геометрий, входных файлов ORCA, логов, итоговых `.csv`, `.xyz`, графиков и отчетов.

## Что делает проект

Проект автоматизирует полный цикл поиска устойчивых структур `B12`:

1. генерирует регулярные и случайные стартовые геометрии;
2. запускает массовый скрининг на `GFN2-xTB`;
3. отбирает сошедшиеся и структурно уникальные минимумы;
4. переоптимизирует лучшие кандидаты методом `r2SCAN-3c`;
5. проверяет их на отсутствие мнимых частот расчетом `NumFreq`;
6. собирает итоговые таблицы и отчеты.

## Текущий результат по данным репозитория

Для текущего набора расчетов получены следующие финальные структуры:

| Ранг | Структура | Мультиплетность | Энергия DFT, Eh | ΔE, кДж/моль | Мин. частота, см^-1 | Статус |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `double_ring_6_6_d1.90_m1_r2scan` | 1 | -297.805239027148 | 0.00 | 0.0 | `OK_NO_IMAG` |
| 2 | `ring12_d1.70_m1_r2scan` | 1 | -297.564484568076 | 632.10 | 0.0 | `OK_NO_IMAG` |
| 3 | `zigzag_chain12_d1.50_m3_r2scan` | 3 | -297.451682484110 | 928.26 | -2022.99 | `REJECT_IMAG` |

Главный вывод по текущему расчету: наиболее стабильной структурой в исследованной выборке является двухкольцевой синглетный мотив `double_ring_6_6`.

## Ключевые параметры

Основные настройки заданы в [`settings.json`](settings.json):

- заряд: `0`
- число атомов: `12`
- мультиплетности: `1`, `3`
- стартовые расстояния: `1.5`, `1.7`, `1.9`, `2.1 A`
- число случайных стартов на каждое расстояние: `10`
- xTB-этап: `GFN2-xTB TightSCF TightOpt`
- DFT-этап: `r2SCAN-3c TightSCF TightOpt`
- частотный этап: `r2SCAN-3c TightSCF NumFreq`
- порог значимой мнимой частоты: `-20.0 см^-1`

## Схема пайплайна

### 1. Генерация стартов

```bash
python3 pipeline.py make-starts
```

Создает:

- регулярные мотивы `ring12`, `double_ring_6_6`, `zigzag_chain12`
- случайные компактные структуры для нескольких стартовых расстояний

### 2. Подготовка и запуск xTB-скрининга

```bash
python3 pipeline.py make-xtb-inputs
./run_jobs.sh jobs/01_xtb
```

### 3. Ранжирование xTB и удаление дублей

```bash
python3 pipeline.py rank-xtb
```

На этом этапе формируются:

- [`results/xtb_ranked_all.csv`](results/xtb_ranked_all.csv)
- [`results/xtb_ranked_unique.csv`](results/xtb_ranked_unique.csv)
- [`results/xtb_top_for_dft.csv`](results/xtb_top_for_dft.csv)

### 4. Подготовка и запуск DFT-уточнения

```bash
python3 pipeline.py make-dft-inputs
./run_jobs.sh jobs/02_r2scan
```

### 5. Подготовка и запуск частотного анализа

```bash
python3 pipeline.py make-numfreq-inputs
./run_jobs.sh jobs/03_numfreq
```

### 6. Сбор финального отчета

```bash
python3 pipeline.py final-report
```

Формируются:

- [`results/final_dft_ranked.csv`](results/final_dft_ranked.csv)
- [`results/final_numfreq_report.csv`](results/final_numfreq_report.csv)
- [`results/final_good_no_imag.csv`](results/final_good_no_imag.csv)

## Структура репозитория

```text
.
├── jobs/
│   ├── 01_xtb/
│   ├── 02_r2scan/
│   └── 03_numfreq/
├── results/
│   ├── 01_xtb_xyz/
│   ├── 02_r2scan_xyz/
│   ├── xtb_ranked_all.csv
│   ├── xtb_ranked_unique.csv
│   ├── final_dft_ranked.csv
│   ├── final_numfreq_report.csv
│   ├── final_good_no_imag.csv
│   ├── calculation_report_ru.md
│   └── academic_report_ru.md
├── starts/
├── plotly_charts/
├── seaborn_charts/
├── numfreq_charts/
├── pipeline.py
├── run_jobs.sh
└── settings.json
```

## Основные артефакты

### Отчеты

- [Краткий расчетный отчет](results/calculation_report_ru.md)
- [Академический отчет](results/academic_report_ru.md)

### Итоговые таблицы

- [Уникальные минимумы после xTB](results/xtb_ranked_unique.csv)
- [Финальные DFT-структуры](results/final_dft_ranked.csv)
- [Частотный отчет](results/final_numfreq_report.csv)
- [Структуры без мнимых частот](results/final_good_no_imag.csv)

### Геометрии

- [xTB-геометрии](results/01_xtb_xyz/)
- [DFT-геометрии](results/02_r2scan_xyz/)

### Визуализации

- [Plotly dashboard](plotly_charts/out/index.html)
- [Seaborn charts](seaborn_charts/out/index.html)
- [NumFreq charts](numfreq_charts/out/index.html)

## Требования

Для воспроизведения расчетов необходимы:

- Linux или VPS-среда
- установленный ORCA 6.1
- доступный путь к бинарнику ORCA в `settings.json`
- Python 3 для запуска `pipeline.py`
- shell-скрипт `run_jobs.sh`

Бинарник ORCA в репозиторий не входит.

## Что важно для диплома или статьи

Если репозиторий используется как основа для дипломной работы, в итоговом тексте имеет смысл акцентировать:

- многостартовый характер поиска, а не одну стартовую геометрию;
- автоматическое удаление структурных дублей;
- переход от дешевого `GFN2-xTB` к `r2SCAN-3c`;
- обязательную частотную верификацию минимумов;
- финальное выделение структур без мнимых частот.

## Примечания

- В `results/final_good_no_imag.csv` остаются только структуры, у которых `n_imag_strict == 0`.
- В `results/xtb_ranked_all.csv` присутствуют две технические тестовые записи `test_xtb_fix` и `test_xtb_fix2`; основной расчетный набор состоит из `104` рабочих xTB-задач.
- Файл [`template_goat_explore.inp`](template_goat_explore.inp) можно использовать как дополнительную заготовку для ручного GOAT-EXPLORE, но основной репозиторный workflow построен на многостартовом скрининге и DFT-уточнении.
