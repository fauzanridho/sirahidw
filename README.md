# SIRAH IDW

**(SIRAH IDW) Sistem Interpolasi Curah Hujan Menggunakan IDW dan LOOCV** adalah aplikasi Streamlit untuk mengolah data curah hujan berbasis titik stasiun, melakukan akumulasi harian, bulanan, atau tahunan, mencari parameter Inverse Distance Weighting (IDW) terbaik dengan Leave-One-Out Cross Validation (LOOCV), lalu membuat hasil interpolasi titik, grid, dan raster.

## Format Data

Siapkan data CSV atau XLSX dalam format long dengan kolom minimal:

```csv
Tanggal,Nama_Stasiun,Longitude,Latitude,Curah_Hujan
2025-01-01,Waduk Nawangan,110.897683,-8.041533,2.84
2025-01-01,Giriwoyo,110.947632,-8.026094,10
```

Kolom wajib:

- `Tanggal`
- `Nama_Stasiun`
- `Longitude`
- `Latitude`
- `Curah_Hujan`

File batas wilayah bersifat opsional dan dapat berupa ZIP SHP, beberapa komponen SHP (`.shp`, `.shx`, `.dbf`, `.prj`), atau GeoJSON.

Data curah hujan yang diupload atau diinput manual disimpan permanen ke:

```text
data/curah_hujan_master.csv
```

Jika data bertambah, upload file baru atau isi input manual. Aplikasi dapat menambahkan baris baru, memperbarui baris duplikat berdasarkan `Tanggal`, `Nama_Stasiun`, `Longitude`, dan `Latitude`, atau mengganti seluruh master data.

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

## Alur Penggunaan

1. Buka menu **Upload Data** untuk mengunggah CSV/XLSX atau menambahkan data lewat **Input Manual**.
2. Jika tersedia, unggah batas wilayah ZIP SHP, komponen SHP, atau GeoJSON.
3. Masuk ke **Pilih Data** untuk memilih data dari master permanen berdasarkan stasiun dan rentang tanggal.
4. Masuk ke **Pra-pemrosesan** untuk membersihkan kolom, mengubah tipe data, dan melihat statistik data.
5. Masuk ke **Akumulasi Data**, pilih skala **Harian**, **Bulanan**, atau **Tahunan**.
6. Pilih metode agregasi. Default aplikasi adalah **SUM** karena curah hujan merupakan data akumulatif.
7. Pilih periode analisis: tanggal untuk harian, bulan dan tahun untuk bulanan, atau tahun untuk tahunan.
8. Masuk ke **LOOCV Tuning IDW**, isi daftar nilai `p` dan `k`, lalu jalankan LOOCV.
9. Gunakan parameter terbaik dari LOOCV pada **Interpolasi IDW Final** untuk prediksi titik manual, titik target CSV, atau data missing.
10. Masuk ke **Peta Grid/Raster** untuk membuat peta titik, grid interpolasi, PNG, CSV grid, dan GeoTIFF.
11. Ambil seluruh output pada menu **Unduh Hasil**.

## Output

- Tabel statistik pra-pemrosesan.
- Tabel data periode terpilih.
- Tabel hasil LOOCV berisi `p`, `k`, `MAE`, `RMSE`, `MAPE`, dan jumlah data valid.
- Grafik RMSE/MAE, observasi vs prediksi, error per stasiun, dan heatmap parameter.
- Tabel hasil estimasi IDW.
- Peta titik stasiun berbasis Folium.
- Peta grid interpolasi berbasis Matplotlib.
- GeoTIFF raster IDW dengan CRS default `EPSG:4326` dan `NoData = -9999`.
