import sys
import rasterio
from rasterio.transform import xy
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QGraphicsView, QGraphicsScene, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import QRectF, Qt
import numpy as np

class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.start = None
        self.rect_item = None
        self.transform_geo = None
        self.scale_factor = 1.0  # Jak bardzo pomniejszyliśmy oryginał

    def set_image(self, img_array, transform, scale_factor):
        self.scene.clear()
        self.transform_geo = transform
        self.scale_factor = scale_factor

        #Upewniamy się, że dane są ułożone ciągle w pamięci
        img_array = np.ascontiguousarray(img_array)

        h, w, c = img_array.shape
        bytes_per_line = c * w

        #rzutujemy na odpowiedni typ, aby QImage go zaakceptował
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self._image_ref = img_array # Musimy zachować referencję do img_array, bo QImage nie kopiuje danych, tylko operuje na wskaźniku. Jeśli Python usunie img_array, program się wywali.
       
        pixmap = QPixmap.fromImage(qimg)
        self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(0, 0, w, h))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        self.start = self.mapToScene(event.pos())
        if self.rect_item:
            self.scene.removeItem(self.rect_item)
        self.rect_item = self.scene.addRect(QRectF(self.start, self.start), QColor(255, 0, 0))
        self.rect_item.setBrush(QColor(255, 0, 0, 50))

    def mouseMoveEvent(self, event):
        if self.start:
            current = self.mapToScene(event.pos())
            self.rect_item.setRect(QRectF(self.start, current).normalized())

    def mouseReleaseEvent(self, event):
        end = self.mapToScene(event.pos())
        
        # Przeliczamy współrzędne z widoku (podglądu) na oryginalne piksele
        orig_x1, orig_y1 = self.start.x() * self.scale_factor, self.start.y() * self.scale_factor
        orig_x2, orig_y2 = end.x() * self.scale_factor, end.y() * self.scale_factor

        # pixel -> geo (używając transformacji ORYGINALNEGO pliku)
        x_geo1, y_geo1 = xy(self.transform_geo, int(orig_y1), int(orig_x1))
        x_geo2, y_geo2 = xy(self.transform_geo, int(orig_y2), int(orig_x2))

        bbox = (min(x_geo1, x_geo2), min(y_geo1, y_geo2), max(x_geo1, x_geo2), max(y_geo1, y_geo2))
        print(f"BBOX GEO (Oryginalne współrzędne): {bbox}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tree Detection - Cropper")
        self.view = ImageView()
        self.paths = {
        "orto": None, 
        "nmt": None, 
        "nmpt": None
    }
        self.btn_load = QPushButton("Wczytaj ortofoto")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_nmpt=QPushButton("Wczytaj nmpt")
        
        self.btn_nmt = QPushButton("Wczytaj nmt")
        self.btn_nmt.clicked.connect(lambda: self.set_path('nmt'))
        self.btn_start = QPushButton("Start")
        self.btn_nmpt.clicked.connect(lambda: self.set_path('nmpt'))

                
        layout = QVBoxLayout()
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_nmpt)
        layout.addWidget(self.btn_nmt)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.view)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(1000, 800)
    
    def set_path(self, key):
    # Otwiera okno wyboru pliku
        path, _ = QFileDialog.getOpenFileName(self, f"Wybierz {key.upper()}", "", "Asc (*.asc)")
    
        if path:
            self.paths[key] = path  # ZAPAMIĘTANIE: tutaj ścieżka trafia do zmiennej

            
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wybierz GeoTIFF", "", "TIFF Files (*.tif *.tiff)")
        if not path: return

        with rasterio.open(path) as src:
            # 1. Obliczamy współczynnik pomniejszenia (np. max 2000px szerokości dla podglądu)
            max_dim = 2000
            scale = max(src.width, src.height) / max_dim
            if scale < 1: scale = 1
            
            new_width = int(src.width / scale)
            new_height = int(src.height / scale)

            # 2. Wczytujemy pomniejszony obraz (Thumbnail) - OSZCZĘDZA RAM I CZAS
            img = src.read(
                [1, 2, 3],
                out_shape=(3, new_height, new_width),
                resampling=rasterio.enums.Resampling.bilinear
            )
            
            img = np.transpose(img, (1, 2, 0))
            # Prosta normalizacja dla wizualizacji
            img = np.clip(img, 0, 255).astype(np.uint8) 

            self.view.set_image(img, src.transform, scale)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())