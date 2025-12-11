<!-- f06845b5-0cda-4d4e-92ad-da2473924f71 c3d4218f-531c-468a-b216-4296d2ed34f9 -->
# Implementasi Quarterly Aggregation dan Borda Count Voting

## Analisis Sistem Saat Ini

### Flow Data Forecasting

File [`forecast.py`](forecast.py) menghasilkan forecast 24 bulan dengan flow:

1. Membaca data Excel → `read_excel_latest()`
2. Agregasi bulanan → `monthly_aggregate()`
3. Forecast per produk → `forecast_for_product()` atau `fallback_forecast()`
4. Top N per bulan → sorting by "mean" descending

### Struktur Output Saat Ini

- `topN_per_month_24m.csv`: kolom `date, rank, product, category, mean`
- Periode: Mei 2025 - April 2027 (24 bulan)
- Method ranking: simple sort by forecast mean value

### Dependencies

- pandas, numpy, matplotlib, tensorflow.keras
- Type hints sudah digunakan

---

## Rencana Implementasi

### 1. Fungsi `aggregate_to_quarterly()`

```python
def aggregate_to_quarterly(
    monthly_df: pd.DataFrame,
    top_n: int = 5
) -> Dict[str, pd.DataFrame]
```

**Logic:**

- Input: DataFrame hasil forecast bulanan (`forecast_per_product_24m.csv`)
- Mapping bulan ke kuartal: Jan-Mar → Q1, Apr-Jun → Q2, Jul-Sep → Q3, Oct-Dec → Q4
- Agregasi: sum forecast "mean" untuk 3 bulan per kuartal per produk
- Output: Dictionary dengan key tahun (2025, 2026), value DataFrame top 5 per kuartal

### 2. Fungsi `borda_count_ranking()`

```python
def borda_count_ranking(
    quarterly_rankings: Dict[str, pd.DataFrame],
    year: int,
    top_n: int = 5
) -> pd.DataFrame
```

**Logic:**

- Skor Borda: Rank 1 = 5 poin, Rank 2 = 4 poin, ..., Rank 5 = 1 poin
- Produk tidak masuk top 5 di kuartal tersebut = 0 poin
- Agregasi skor dari Q1-Q4
- Handling tie: gunakan total forecast value sebagai tiebreaker
- Output: DataFrame dengan kolom `prod`
- 
- `uct, total_score, rank, score_breakdown`

### 3. Fungsi Visualisasi

#### `plot_yearly_top5()`

- Horizontal bar chart menampilkan top 5 produk tahunan
- Label: nama produk, skor Borda, ranking
- Warna konsisten per produk
- Output: `forecast_plots/top5_yearly_{year}.png`

#### `plot_quarterly_top5()`

- Figure dengan 4 subplots (2x2 grid)
- Setiap subplot: bar chart top 5 produk per kuartal
- Consistent color scheme across subplots
- Output: `forecast_plots/top5_quarterly_{year}.png`

#### `plot_borda_process()` (Optional)

- Stacked horizontal bar chart
- Menampilkan kontribusi skor dari setiap kuartal
- Hanya untuk top 5 produk final
- Output: `forecast_plots/borda_count_process_{year}.png`

### 4. Integrasi ke `forecast.py`

Menambahkan fungsi `forecast`:

1. Panggil `aggregate_to_quarterly()` dengan data dari CSV
2. Panggil `borda_count_ranking()` untuk setiap tahun
3. Generate semua plot
4. Simpan hasil agregasi ke CSV tambahan

---

## Handling Edge Cases

| Case | Solusi |

|------|--------|

| Produk tidak masuk top 5 di semua kuartal | Skor = 0, tidak masuk ranking tahunan |

| Tie dalam Borda Count | Secondary sort by total forecast value (descending) |

| Data tidak lengkap untuk 4 kuartal | Hitung dari kuartal yang tersedia |

| Periode tidak mulai dari Q1 | Mapping dinamis berdasarkan bulan aktual |

---

## Output Files Baru

```
forecast_plots/
├── top5_yearly_2025.png
├── top5_yearly_2026.png
├── top5_quarterly_2025.png
├── top5_quarterly_2026.png
├── borda_count_process_2025.png
└── borda_count_process_2026.png

(CSV tambahan)
├── quarterly_top5_2025.csv
├── quarterly_top5_2026.csv
├── yearly_top5_borda_2025.csv
└── yearly_top5_borda_2026.csv
```

---

## Catatan Teknis

- Semua plot disimpan dengan DPI 300
- Folder dibuat otomatis dengan `os.makedirs(..., exist_ok=True)`
- Logging ditambahkan untuk tracking proses
- Type hints dan docstrings untuk semua fungsi baru

### To-dos

- [ ] Implementasi fungsi aggregate_to_quarterly() untuk grouping data bulanan ke kuartal
- [ ] Implementasi fungsi borda_count_ranking() dengan Borda Count Voting algorithm
- [ ] Implementasi fungsi plot_yearly_top5() untuk visualisasi top 5 tahunan
- [ ] Implementasi fungsi plot_quarterly_top5() untuk visualisasi top 5 per kuartal
- [ ] Implementasi fungsi plot_borda_process() untuk visualisasi kontribusi skor (optional)
- [ ] implementasi semua fungsi baru ke file forecast.py