# 🤟 TemanIsyarat - Sistem Pengenalan SIBI (Sistem Isyarat Bahasa Indonesia)

**TemanIsyarat** adalah proyek berbasis Machine Learning & Computer Vision yang dirancang untuk mengenalkan dan menerjemahkan bahasa isyarat SIBI (Sistem Isyarat Bahasa Indonesia) secara *real-time* menggunakan **MediaPipe** dan **TensorFlow/Keras**.

---

## 📌 Fitur Utama

- ✋ **Ekstraksi Landmark Tangan (MediaPipe Hands)**: Ekstraksi 21 koordinat landmark 3D (63 fitur x, y, z) yang dinormalisasi secara relatif terhadap pergelangan tangan (wrist).
- 🧠 **Model Deep Learning (TensorFlow/Keras)**: Model Neural Network teroptimasi untuk mengklasifikasikan abjad bahasa isyarat SIBI (A–Y).
- ⚡ **Konversi TFLite**: Model yang dikonversi ke format `.tflite` agar efisien dan ringan untuk dijalankan pada perangkat *edge* atau mobile (Android/Flutter).
- 📹 **Pengujian Real-Time Webcam (OpenCV)**: Pengujian langsung pergerakan tangan dan prediksi abjad melalui kamera web dengan *confidence score*.

---

## 📁 Struktur Direktori Proyek

```text
TemanIsyarat/
├── python/
│   ├── dataset/                      # Folder dataset gambar (diabaikan dari Git)
│   ├── dataset_sibi_landmarks.csv    # CSV hasil ekstraksi landmark 63 titik
│   ├── extraction.py                 # Script ekstraksi fitur gambar ke CSV
│   ├── extraction_new.py             # Script ekstraksi alternatif/terbaru
│   ├── train.py                      # Script pelatihan model Keras (.h5)
│   ├── train_new.py                  # Script pelatihan model alternatif
│   ├── convert.py                    # Script konversi Keras (.h5) ke TFLite (.tflite)
│   ├── test.py                       # Uji coba real-time webcam dengan TFLite
│   ├── test_new.py                   # Script pengujian alternatif
│   ├── sibi_model.h5                 # Model terlatih Keras
│   └── sibi_model.tflite             # Model teroptimasi TFLite
├── .gitignore                        # Konfigurasi pengabaian file Git
└── README.md                         # Dokumentasi proyek
```

---

## 🛠️ Teknologi & Library

- **Python 3.8+**
- **OpenCV**: Pengolahan citra & pemrosesan stream webcam.
- **MediaPipe**: Deteksi & pelacakan 21 koordinat *hand landmark*.
- **TensorFlow / Keras**: Pembangunan & pelatihan model Deep Learning.
- **Scikit-Learn**: *Label encoding* dan *train-test split*.
- **Pandas & NumPy**: Manipulasi data & komputasi matriks.

---

## 🚀 Panduan Penggunaan

### 1. Persiapan Environment

Install dependensi yang diperlukan terlebih dahulu:

```bash
pip install opencv-python mediapipe tensorflow scikit-learn pandas numpy
```

---

### 2. Ekstraksi Fitur Landmark

Jika Anda memiliki dataset gambar di dalam folder `python/dataset/SIBI/`, jalankan ekstraksi fitur untuk menghasilkan file CSV landmark:

```bash
cd python
python extraction.py
```

---

### 3. Pelatihan Model (Training)

Latih model Neural Network menggunakan file CSV hasil ekstraksi:

```bash
python train.py
```
*Hasil model akan disimpan sebagai `sibi_model.h5`.*

---

### 4. Konversi ke TensorFlow Lite (TFLite)

Konversikan model `.h5` menjadi file `.tflite`:

```bash
python convert.py
```
*Hasil konversi akan disimpan sebagai `sibi_model.tflite`.*

---

### 5. Uji Coba Real-Time (Webcam)

Jalankan script pengujian untuk mendeteksi isyarat tangan SIBI secara langsung dari kamera:

```bash
python test.py
```
*Tekan tombol **'q'** untuk keluar dari jendela webcam.*

---

## 🏷️ Abjad SIBI yang Didukung

Model ini melatih dan mendeteksi abjad alfabet SIBI:
> `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`

---

## 📄 Lisensi

Proyek ini dibuat untuk tujuan edukasi dan pengembangan teknologi aksesibilitas.
