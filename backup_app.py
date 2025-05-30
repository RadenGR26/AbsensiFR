import os
import cv2
import numpy as np
import time
import json
from datetime import datetime
from io import BytesIO
import paho.mqtt.client as mqtt
from flask import Flask, render_template, request, redirect, url_for, Response, flash, session, send_file

# --- Konfigurasi Aplikasi ---
DATA_DIR = "data/known_faces_opencv"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = "deteksiwajah.json"

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Konfigurasi MQTT
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "facerecognition/doorlock/open"

REGISTER_STATE = {}


# --- Fungsi Utilitas ---

def _setup_mqtt_client():
    """Menyiapkan koneksi ke broker MQTT."""
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()
    return client


def log_detection(name: str, confidence: float, accuracy: float):  # Tambahkan parameter accuracy
    """Menambahkan catatan deteksi wajah ke file JSON."""
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "confidence": round(confidence, 2),
        "accuracy_percent": round(accuracy, 2)  # Tambahkan akurasi ke log
    }
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def gen_register(name, num_samples=50, delay=0.7):
    """
    Menghasilkan frame video untuk proses registrasi wajah.
    Mengambil 'num_samples' gambar wajah untuk setiap nama.
    """
    save_dir = os.path.join(DATA_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0
    last_capture = time.time()

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample {count + 1}/{num_samples}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            if time.time() - last_capture > delay:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (200, 200))

                face_img = cv2.equalizeHist(face_img)

                img_path = os.path.join(save_dir, f"{name}_{count + 1}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1
                last_capture = time.time()
                if count >= num_samples:
                    break

        cv2.putText(frame, f"Capturing face: {count}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()
    REGISTER_STATE[name] = True


def prepare_training_data(data_folder_path):
    """Mempersiapkan data gambar wajah untuk pelatihan model."""
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for name in os.listdir(data_folder_path):
        person_dir = os.path.join(data_folder_path, name)
        if not os.path.isdir(person_dir):
            continue

        label_map[current_label] = name

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(current_label)
        current_label += 1
    return faces, labels, label_map


def gen_recognition():
    """
    Menghasilkan frame video untuk proses pengenalan wajah.
    Mengenali wajah, menampilkan nama, Jarak Wajah (Confidence),
    dan Akurasi Kemiripan (persentase).
    """
    total_registered_images = 0
    for name in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, name)
        if os.path.isdir(person_dir):
            total_registered_images += len(os.listdir(person_dir))

    MIN_SAMPLES_FOR_RECOGNITION = 50
    if total_registered_images < MIN_SAMPLES_FOR_RECOGNITION:
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(frame, f"Data wajah belum cukup ({total_registered_images}/{MIN_SAMPLES_FOR_RECOGNITION} sampel)",
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    faces, labels, label_map = prepare_training_data(DATA_DIR)

    if len(faces) == 0:
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(frame, "Belum ada data wajah! Silahkan daftar.", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    mqtt_client = _setup_mqtt_client()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    mqtt_sent = False

    # --- Ambang Batas Confidence untuk penentuan Dikenali/Tidak Dikenali ---
    # Jika confidence < 50, maka wajah DIKENALI
    # Jika confidence >= 50, maka wajah TIDAK DIKENALI
    CONFIDENCE_THRESHOLD = 50
    # --- Akhir Konfigurasi ---

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        recognized_this_frame = False
        recognized_name = None  # Deklarasi di sini
        current_confidence = None
        current_accuracy = None

        for (x, y, w, h) in faces_rects:
            face_roi = gray[y:y + h, x:x + w]

            face_roi_equalized = cv2.equalizeHist(face_roi)

            face_roi_resized = cv2.resize(face_roi_equalized, (faces[0].shape[1], faces[0].shape[0]))

            label, confidence = recognizer.predict(face_roi_resized)

            # --- Hitung Akurasi Kemiripan (dibalik dari Confidence) ---
            # Max confidence bisa bervariasi, tapi 100 adalah asumsi yang baik untuk pembatasan visual.
            # Jika confidence 0, akurasi 100%. Jika confidence 100, akurasi 0%.
            accuracy_percent = max(0, 100 - confidence)
            # --- Akhir Perhitungan Akurasi Kemiripan ---

            # --- LOGIKA PENENTUAN DAN TAMPILAN SESUAI PERMINTAAN ---
            if confidence < CONFIDENCE_THRESHOLD:  # Akurasi kemiripan di atas 50%
                name_display = label_map.get(label, "Unknown")
                status_text = f"Label: {name_display}"
                confidence_text = f"Jarak Wajah (Confidence): {confidence:.0f} (10-50)"
                accuracy_text = f"Akurasi Kemiripan: {accuracy_percent:.0f}% (di atas 50%)"
                color = (0, 255, 0)  # Hijau untuk Dikenali

                recognized_this_frame = True
                recognized_name = name_display
                current_confidence = confidence
                current_accuracy = accuracy_percent

            else:  # Akurasi kemiripan di bawah 50%
                name_display = "Tidak Dikenali"
                status_text = f"Label: {name_display}"
                confidence_text = f"Jarak Wajah (Confidence): {confidence:.0f} (50-100+)"
                accuracy_text = f"Akurasi Kemiripan: {accuracy_percent:.0f}% (di bawah 50%)"
                color = (0, 0, 255)  # Merah untuk Tidak Dikenali

            # Tampilkan informasi di atas kotak deteksi
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, status_text, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, confidence_text, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, accuracy_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # --- Akhir LOGIKA PENENTUAN DAN TAMPILAN ---

        # Pastikan variabel memiliki nilai sebelum digunakan
        if recognized_this_frame and recognized_name is not None and current_confidence is not None and current_accuracy is not None:
            if not mqtt_sent:
                mqtt_client.publish(MQTT_TOPIC, payload=recognized_name)
                log_detection(recognized_name, current_confidence, current_accuracy)  # Log juga akurasi
                mqtt_sent = True
        else:
            mqtt_sent = False

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()


# --- Routing Flask ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name", "").strip()
    if not name:
        flash("Nama tidak boleh kosong.")
        return redirect(url_for('index'))
    REGISTER_STATE[name] = False
    session['register_name'] = name
    return redirect(url_for('register_live'))


@app.route("/register_live")
def register_live():
    name = session.get('register_name')
    if not name:
        flash("Mohon masukkan nama terlebih dahulu.")
        return redirect(url_for('index'))
    return render_template("register_live.html", name=name)


@app.route("/register_feed")
def register_feed():
    name = session.get('register_name')
    if not name:
        return "Nama tidak ada", 400
    if REGISTER_STATE.get(name):
        return Response(status=204)
    return Response(gen_register(name), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/register_done")
def register_done():
    name = session.get('register_name')
    if name:
        flash(f"Registrasi untuk {name} selesai dan berhasil disimpan.")
    return redirect(url_for('index'))


@app.route("/recognize")
def recognize():
    return render_template("recognize.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/records')
def records():
    """Menampilkan 10 catatan deteksi wajah terakhir."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    last10 = data[-10:] if data else []
    last10.reverse()
    return render_template('records.html', records=last10)

@app.route('/clear_records')
def clear_records():
    """Menghapus seluruh file log deteksi wajah."""
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
            flash("Seluruh data riwayat deteksi berhasil dihapus.")
        except Exception as e:
            flash(f"Gagal menghapus data riwayat: {e}")
    else:
        flash("Tidak ada data riwayat untuk dihapus.")
    return redirect(url_for('records'))

@app.route('/export')
def export():
    """Mengizinkan ekspor data deteksi wajah ke file Excel."""
    try:
        import pandas as pd
    except ImportError:
        flash("Mohon instal pandas terlebih dahulu: pip install pandas")
        return redirect(url_for('index'))

    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_json(LOG_FILE)
        except Exception:
            df = pd.DataFrame(columns=["time", "name", "confidence", "accuracy_percent"])
    else:
        df = pd.DataFrame(columns=["time", "name", "confidence", "accuracy_percent"])

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='DeteksiWajah')
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="deteksiwajah.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    app.run(debug=True)