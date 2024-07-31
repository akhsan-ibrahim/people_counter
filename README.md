
# People Counter System

People Counter System adalah proyek yang bertujuan untuk menghitung jumlah orang yang melewati suatu area tertentu menggunakan teknologi deteksi dan pelacakan objek. Sistem ini dibangun dengan menggunakan kamera yang dipasang pada jalur keluar-masuk dan diintegrasikan dengan framework web Flask untuk memantau hasil penghitungan secara real-time.
## Teknologi yang Digunakan

- **Backend**: Flask, Python 3.11
- **Computer Vision**: OpenCV, [YOLOv8](https://github.com/ultralytics/ultralytics), [DeepSORT](https://github.com/computervisioneng/object-tracking-yolov8-deep-sort)
- **Frontend**: HTML
## Perangkat yang Digunakan

| Komponen  | Keterangan           |
| :-------- | :------------------- |
| OS        | Windows 11 Pro       |
| CPU       | Intel i7-12650H      |
| RAM       | 16 GB                |
| GPU       | RTX 4060 (CUDA 11.8) |
| CCTV      | Hikvision            |

## Instalasi

1. Clone Repository

```bash
  git clone https://github.com/akhsan-ibrahim/people_counter.git
  cd people-counter-system
```

2. Buat Virtual Environment dan Instal Dependensi

```bash
  python -m venv venv
  source venv/bin/activate   # untuk pengguna Unix atau MacOS
  venv\Scripts\activate      # untuk pengguna Windows
  pip install -r requirements.txt
```

3. Pilih Sumber (Live/Rekaman)
Pada berkas **camera.py**.

```bash
  class Camera():
    def __init__(self):
      self.live = True      # ubah menjadi "False" untuk menggunakan video rekaman
      self.cap = self.source(self.live)
```
4. Ubah Tautan Sumber
Ganti tautan **rtsp** sesuai konfigurasi kamera live.

```bash
  def source(self, live):
      cap = cv2.VideoCapture("resources/Gambar 5.mp4")
      if live == True:
         cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.195.167:554/Streaming/Channels/501")
      return cap
```

5. Jalankan Aplikasi

```bash
  python main.py
```

6. Akses Antarmuka Web -
Buka browser dan navigasikan ke http://localhost:5000 untuk mengakses antarmuka web.