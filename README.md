import cv2
import urllib.request
import os

# Nama file model Haar Cascade
face_cascade_filename = "haarcascade_frontalface_default.xml"
face_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

def download_haarcascade():
    """Mengunduh file haarcascade jika tidak ada"""
    if not os.path.exists(face_cascade_filename):
        print("Mengunduh model Haar Cascade...")
        try:
            urllib.request.urlretrieve(face_cascade_url, face_cascade_filename)
            print("Pengunduhan selesai.")
        except Exception as e:
            print(f"Error saat mengunduh: {e}")
            return False
    return True

def process_camera():
    """Fungsi untuk memproses input dari kamera"""
    if not download_haarcascade():
        print("Gagal mengunduh file Haar Cascade. Program dihentikan.")
        return

    face_cascade = cv2.CascadeClassifier(face_cascade_filename)

    cap = cv2.VideoCapture(0)  # 0 = kamera default

    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Deteksi Wajah (Kamera)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Pemrosesan kamera selesai.")

if __name__ == "__main__":
    process_camera()
