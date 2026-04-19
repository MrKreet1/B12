# ORCA 6.1: B12-кластер на VPS (8 CPU / 24 GB RAM)

Этот набор файлов делает рабочий конвейер под Linux/VPS для задачи:

1. сгенерировать много стартовых геометрий B12;
2. прогнать дешёвую оптимизацию на `GFN2-xTB`;
3. отобрать **уникальные** низкоэнергетические минимумы;
4. переоптимизировать лучшие структуры на `r2SCAN-3c`;
5. проверить частоты (`NumFreq`);
6. получить финальный отчёт, где отдельно выделены структуры **без мнимых частот**.

## Что важно для диплома

Не опирайтесь только на одну стартовую геометрию или на один 1D scan.
Для B12 лучше показывать:

- много стартовых конфигураций;
- отбор уникальных минимумов;
- DFT-переоптимизацию;
- проверку частот;
- итоговую таблицу энергий и отметку, есть ли мнимые частоты.

## Что именно делает пакет

- `pipeline.py make-starts`  
  генерирует:
  - несколько регулярных мотивов (`ring12`, `double_ring_6_6`, `zigzag_chain12`)
  - много случайных компактных стартов
  - всё это для нескольких стартовых расстояний из `seed_distances_ang`

- `pipeline.py make-xtb-inputs`  
  создаёт ORCA input-файлы для первого скрининга на `GFN2-xTB`

- `run_jobs.sh jobs/01_xtb`  
  последовательно запускает все `.inp` из папки

- `pipeline.py rank-xtb`  
  парсит выходы ORCA, извлекает финальные энергии и финальные координаты, 
  удаляет дубликаты через fingerprint на основе отсортированных попарных расстояний,
  пишет:
  - `results/xtb_ranked_all.csv`
  - `results/xtb_ranked_unique.csv`
  - `results/xtb_top_for_dft.csv`

- `pipeline.py make-dft-inputs`  
  делает input-файлы `r2SCAN-3c TightOpt TightSCF`

- `run_jobs.sh jobs/02_r2scan`

- `pipeline.py make-numfreq-inputs`
  берёт лучшие DFT-структуры и создаёт input-файлы для `NumFreq`

- `run_jobs.sh jobs/03_numfreq`

- `pipeline.py final-report`
  собирает итоговый отчёт:
  - `results/final_dft_ranked.csv`
  - `results/final_numfreq_report.csv`
  - `results/final_good_no_imag.csv`

## Как пользоваться

### 1) Отредактируйте настройки

Откройте `settings.json` и проверьте:

- `orca_path` — путь к бинарнику ORCA
- `charge`
- `multiplicities`
- `seed_distances_ang`
- `top_n_xtb_for_dft`
- `top_n_dft_for_numfreq`

Пример пути:
```bash
/home/gnom_ekubo/apps/orca_6_1_1/orca
```

### 2) Сделайте скрипт запуска исполняемым
```bash
chmod +x run_jobs.sh
```

### 3) Сгенерируйте старты
```bash
python3 pipeline.py make-starts
```

### 4) Подготовьте xTB-скрининг
```bash
python3 pipeline.py make-xtb-inputs
./run_jobs.sh jobs/01_xtb
```

### 5) Отберите лучшие структуры
```bash
python3 pipeline.py rank-xtb
```

### 6) Подготовьте DFT-переоптимизацию
```bash
python3 pipeline.py make-dft-inputs
./run_jobs.sh jobs/02_r2scan
```

### 7) Подготовьте частоты
```bash
python3 pipeline.py make-numfreq-inputs
./run_jobs.sh jobs/03_numfreq
```

### 8) Соберите итоговый отчёт
```bash
python3 pipeline.py final-report
```

## Как контролируется требование “без мнимых чисел”

В `final_numfreq_report.csv` есть два счётчика:

- `n_imag_strict` — число частот `< 0.0 cm^-1`
- `n_imag_significant` — число частот `< imag_threshold_significant_cm1`

Итоговый файл:
- `final_good_no_imag.csv`

содержит **только структуры, у которых `n_imag_strict == 0`**.

То есть файл уже отфильтрован по вашему жёсткому условию:
**никаких мнимых частот**.

## Что делать, если появились мнимые частоты

Если даже у лучших структур есть мнимые частоты:

1. не брать их в финальный вывод;
2. увеличить качество геометрической оптимизации;
3. перезапустить `r2SCAN-3c TightOpt`;
4. при необходимости вручную исказить геометрию вдоль мнимой моды и переоптимизировать.

## Замечание по мультиплетности

Для B12 не фиксируйте только один спин без проверки.
В `settings.json` можно оставить, например:
```json
"multiplicities": [1, 3]
```
Тогда пакет сам создаст jobs и для singlet, и для triplet, а вы потом сравните энергии.

## Опционально: GOAT-EXPLORE

В пакете есть файл `template_goat_explore.inp`.
Это отдельный шаблон, если захотите дополнительно запускать GOAT-EXPLORE вручную.
Основной автоматизированный конвейер пакета построен на **multi-start xTB screening + DFT refinement**, потому что он проще для полного автозапуска на обычном VPS.

