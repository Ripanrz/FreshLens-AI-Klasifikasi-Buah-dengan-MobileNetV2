import os
import numpy as np
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# --- KONFIGURASI ---
app = Flask(__name__)

# Folder untuk menyimpan gambar yang diupload sementara
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model AI (Hanya sekali saat start)
print("Sedang memuat model AI... Mohon tunggu...")
model = load_model('model_buah.h5')
print("✅ Model berhasil dimuat!")

# Label Kelas (HARUS SESUAI URUTAN ALFABET saat training di Colab)
# Cek output: train_generator.class_indices di Colab tadi
CLASS_NAMES = ['Apel', 'Pisang', 'Salak']

def prediksi_gambar(path_gambar):
    # 1. Buka gambar & ubah ukuran ke 224x224 (sesuai MobileNet)
    img = image.load_img(path_gambar, target_size=(224, 224))
    
    # 2. Ubah jadi array numpy
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # 3. Normalisasi (Sama seperti saat training: rescale 1./255)
    x = x / 255.0
    
    # 4. Prediksi
    output = model.predict(x)
    index_max = np.argmax(output)
    confidence = np.max(output) * 100 # Persentase keyakinan
    
    return CLASS_NAMES[index_max], confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    hasil = None
    path_gambar = None
    persentase = 0

    if request.method == 'POST':
        # Cek apakah ada file yang diupload
        if 'file' not in request.files:
            return render_template('index.html', msg="Tidak ada file!")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', msg="Belum pilih gambar!")

        if file:
            # Simpan file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Lakukan Prediksi
            nama_buah, akurasi = prediksi_gambar(file_path)
            
            # Kirim data ke HTML
            hasil = nama_buah
            persentase = round(akurasi, 2)
            path_gambar = file_path

    return render_template('index.html', hasil=hasil, akurasi=persentase, gambar=path_gambar)

if __name__ == '__main__':
    app.run(debug=True, port=5000)