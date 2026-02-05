# 🍏 FreshLens AI: Klasifikasi Buah Real-Time dengan MobileNetV2

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green)
![Flask](https://img.shields.io/badge/Deployment-Flask-lightgrey.svg)
![Bootstrap](https://img.shields.io/badge/Frontend-Bootstrap_5-purple.svg)

Proyek ini mengembangkan sistem **FreshLens AI** untuk mendeteksi jenis buah secara *real-time* dengan tingkat akurasi tinggi.
Sistem ini dirancang menggunakan pendekatan **Transfer Learning** dengan arsitektur **MobileNetV2**, memungkinkan model berjalan ringan namun tetap presisi dalam membedakan buah (Apel, Pisang, Salak) serta memberikan informasi nutrisi terkait.

---

## 📸 Tampilan Antarmuka

![Tampilan Dashboard](FreshLensAI_KlasifikasiBuah_MobileNetV2/Dashboard_klasifikasiBuah.png)

---

## 🚀 Gambaran Umum Sistem

Sistem bekerja melalui **tiga tahapan utama** dalam pipeline pemrosesan citra:

1.  **Tahap 1 – Preprocessing & Augmentasi**
    Citra yang diunggah pengguna diproses ulang:
    - **Resizing:** Mengubah ukuran citra menjadi 224x224 piksel.
    - **Normalization:** Normalisasi nilai piksel (1./255).
    - **Data Augmentation:** (Pada saat training) Rotasi dan pergeseran untuk ketahanan model.

2.  **Tahap 2 – Ekstraksi Fitur (MobileNetV2)**
    Menggunakan *pre-trained model* MobileNetV2 (ImageNet weights) sebagai *feature extractor* untuk mengenali pola visual kompleks (tekstur, bentuk, warna) dari buah tanpa perlu melatih dari nol.

3.  **Tahap 3 – Klasifikasi & Confidence Score**
    Layer klasifikasi kustom (*Custom Head*) menentukan probabilitas kelas:
    - **Apel**
    - **Pisang**
    - **Salak**
    
    Sistem juga menampilkan **Tingkat Keyakinan (Confidence Score)** dalam persentase.

---

## 🔁 Alur Kerja (Prediction Pipeline)

```mermaid
graph LR
    A[Input Gambar] -->|Resize 224x224| B(Augmentasi & Normalisasi)
    B --> C{MobileNetV2 Base}
    C -->|Feature Extraction| D(Global Average Pooling)
    D --> E[Dense Layer 128 + ReLU]
    E --> F[Dropout 0.2]
    F -->|Softmax| G[Output: Label Buah]
