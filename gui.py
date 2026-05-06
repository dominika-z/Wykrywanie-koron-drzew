import rasterio
from rasterio.transform import xy
from rasterio.windows import Window
from deepforest import main
import cv2
import pandas as pd
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QGraphicsView, QGraphicsScene, QVBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import QRectF, QThread, Qt, pyqtSignal
import numpy as np
from shapely import Point, Polygon
import geopandas as gpd
import os

class AnalysisWorker(QThread):
    """Wątek roboczy do analizy DeepForest, aby nie blokować interfejsu GUI."""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)

    def __init__(self, nmt, nmpt, orto, off_x, off_y, width, height):
        super().__init__()
        self.nmt = nmt
        self.nmpt = nmpt
        self.orto = orto
        self.off_x = off_x
        self.off_y = off_y
        self.width = width
        self.height = height

    def run(self):
        self.log_signal.emit("Ładowanie modelu DeepForest...")
        model = main.deepforest()
        model.load_model(model_name="weecology/deepforest-tree", revision="main")

        self.log_signal.emit("Odczytywanie fragmentu ortofotomapy...")
        with rasterio.open(self.orto) as src:
            fragment = Window(self.off_x, self.off_y, self.width, self.height)
            image_rgb = src.read([1, 2, 3], window=fragment).transpose(1, 2, 0)
            crop_transform = src.window_transform(fragment)

        self.log_signal.emit("Przewidywanie koron drzew...")
        predicted = model.predict_image(image=image_rgb.astype("float32"))

        analysis_data = []

        if predicted is not None and not predicted.empty:
            self.log_signal.emit(f"Wykryto {len(predicted)} obiektów. Analiza wysokości i koloru...")
            with rasterio.open(self.nmpt) as nmpt_file, \
                 rasterio.open(self.nmt) as nmt_file:

                nmpt_trans = nmpt_file.transform
                nmt_trans = nmt_file.transform

                for idx, row in predicted.iterrows():
                    print(f"geometria: {row['geometry']}")
                    xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                    tree_crop = image_rgb[ymin:ymax, xmin:xmax]
                    
                    mean_r, mean_g, mean_b = tree_crop.mean(axis=(0, 1))

                    local_x = (xmin + xmax) / 2
                    local_y = (ymin + ymax) / 2
                    world_x, world_y = crop_transform * (local_x, local_y)

                    score = row["score"]

                    p1 = xy(crop_transform, ymin, xmin, offset='ul')
                    p2 = xy(crop_transform, ymin, xmax, offset='ur')
                    p3 = xy(crop_transform, ymax, xmax, offset='lr')
                    p4 = xy(crop_transform, ymax, xmin, offset='ll')

                    coords = [p1, p2, p3, p4, p1]

                    try:
                        val_nmpt = list(nmpt_file.sample([(world_x, world_y)]))[0][0]
                        val_nmt = list(nmt_file.sample([(world_x, world_y)]))[0][0]
                        height = val_nmpt - val_nmt

                    except:
                        print(f"Nie można odczytać wysokości. Ustawiam wysokość na 0.")
                        height = 0

                    analysis_data.append({
                        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                        "local_x": local_x, "local_y": local_y,
                        "mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b,
                        "height": height,
                        "geometryP": Point(float(world_x), float(world_y)),
                        "geometryA": Polygon(coords),
                        "score": score
                    })

            df = pd.DataFrame(analysis_data)

            if not df.empty:
                global_mean_color = df[["mean_r", "mean_g", "mean_b"]].mean().values
                df["color_dist"] = np.linalg.norm(df[["mean_r", "mean_g", "mean_b"]].values - global_mean_color, axis=1)
                median_dist = df["color_dist"].median()
                mad = np.median(np.abs(df["color_dist"] - median_dist))
                clr_threshold = median_dist + 2.5 * mad

                good_trees_mask = (df["height"] > 2.0) & (df["color_dist"] <= clr_threshold)
                good_trees_data = [analysis_data[i] for i in df[good_trees_mask].index]

                os.makedirs("trees", exist_ok=True)
                
                if good_trees_data:
                    gdfA = gpd.GeoDataFrame(
                        good_trees_data,
                        geometry="geometryA",
                        crs=src.crs
                    )
                    gdfA.to_file(r"trees\trees_A.shp")

                    gdfP = gpd.GeoDataFrame(
                        good_trees_data,
                        geometry="geometryP",
                        crs=src.crs
                    )
                    gdfP.to_file(r"trees\trees_P.shp")
                    self.log_signal.emit(f"Zapisano {len(good_trees_data)} dobrych drzew.")
                else:
                    self.log_signal.emit("Brak dobrych drzew spełniających kryteria.")

                display_img = cv2.cvtColor(image_rgb.astype("uint8"), cv2.COLOR_RGB2BGR)

                for _, row in df.iterrows():
                    is_tall_enough = row["height"] > 2.0
                    is_normal_color = row["color_dist"] <= clr_threshold

                    if is_tall_enough and is_normal_color:
                        color = (0, 255, 0)
                    elif not is_tall_enough:
                        color = (255, 0, 0)
                    else:
                        color = (0, 0, 255)

                    cv2.circle(display_img, (int(row["local_x"]), int(row["local_y"])), 15, (0,255,255), -1)
                    cv2.rectangle(display_img, (int(row["xmin"]), int(row["ymin"])), 
                                  (int(row["xmax"]), int(row["ymax"])), color, 8)

                max_width = 1200
                max_height = 800
                h, w = display_img.shape[:2]
                scale = min(max_width / w, max_height / h)
                resized_img = cv2.resize(display_img, (int(w * scale), int(h * scale)))

                self.finished_signal.emit(resized_img)
            else:
                self.log_signal.emit("Brak poprawnych danych z bboxów.")
                self.finished_signal.emit(None)
        else:
            self.log_signal.emit("Nie wykryto żadnych obiektów.")
            self.finished_signal.emit(None)


class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.start = None
        self.rect_item = None
        self.transform_geo = None
        self.scale_factor = 1.0
        self.crop_rect = None

    def set_image(self, img_array, transform, scale_factor):
        self.scene.clear()
        self.transform_geo = transform
        self.scale_factor = scale_factor
        self.crop_rect = None

        img_array = np.ascontiguousarray(img_array)
        h, w, c = img_array.shape
        bytes_per_line = c * w

        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._image_ref = img_array 

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

        orig_x1, orig_y1 = self.start.x() * self.scale_factor, self.start.y() * self.scale_factor
        orig_x2, orig_y2 = end.x() * self.scale_factor, end.y() * self.scale_factor

        x_geo1, y_geo1 = xy(self.transform_geo, int(orig_y1), int(orig_x1))
        x_geo2, y_geo2 = xy(self.transform_geo, int(orig_y2), int(orig_x2))

        bbox = (min(x_geo1, x_geo2), min(y_geo1, y_geo2), max(x_geo1, x_geo2), max(y_geo1, y_geo2))
        print(f"BBOX GEO (Oryginalne współrzędne): {bbox}")
        crop_x = min(orig_x1, orig_x2)
        crop_y = min(orig_y1, orig_y2)
        crop_w = abs(orig_x2 - orig_x1)
        crop_h = abs(orig_y2 - orig_y1)
        self.crop_rect = (int(crop_x), int(crop_y), int(crop_w), int(crop_h))


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
        self.worker = None

        self.btn_load = QPushButton("Wczytaj ortofoto")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_nmpt = QPushButton("Wczytaj nmpt")
        self.btn_nmt = QPushButton("Wczytaj nmt")
        
        self.btn_nmpt.clicked.connect(lambda: self.set_path('nmpt'))
        self.btn_nmt.clicked.connect(lambda: self.set_path('nmt'))

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_analysis)

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
        path, _ = QFileDialog.getOpenFileName(self, f"Wybierz {key.upper()}", "", "Asc (*.asc);;TIFF Files (*.tif *.tiff)")
        if path:
            self.paths[key] = path
            print(f"Załadowano {key.upper()}: {path}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wybierz GeoTIFF", "", "TIFF Files (*.tif *.tiff)")
        if not path: return

        self.paths["orto"] = path
        print(f"Załadowano ORTO: {path}")

        with rasterio.open(path) as src:
            max_dim = 2000
            scale = max(src.width, src.height) / max_dim
            if scale < 1: scale = 1

            new_width = int(src.width / scale)
            new_height = int(src.height / scale)

            img = src.read(
                [1, 2, 3],
                out_shape=(3, new_height, new_width),
                resampling=rasterio.enums.Resampling.bilinear
            )

            img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 255).astype(np.uint8) 

            self.view.set_image(img, src.transform, scale)

    def start_analysis(self):
        if not self.paths["orto"]:
            QMessageBox.warning(self, "Błąd", "Wczytaj najpierw plik ortofotomapy!")
            return
        if not self.paths["nmt"] or not self.paths["nmpt"]:
            QMessageBox.warning(self, "Błąd", "Wczytaj najpierw pliki NMT i NMPT!")
            return
        if not self.view.crop_rect:
            QMessageBox.warning(self, "Błąd", "Zaznacz najpierw obszar myszką na obrazie!")
            return

        off_x, off_y, width, height = self.view.crop_rect
        print(f"Uruchamiam analizę obszaru: X:{off_x}, Y:{off_y}, W:{width}, H:{height}")
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Analiza w toku...")

        self.worker = AnalysisWorker(
            nmt=self.paths["nmt"],
            nmpt=self.paths["nmpt"],
            orto=self.paths["orto"],
            off_x=off_x,
            off_y=off_y,
            width=width,
            height=height
        )

        self.worker.log_signal.connect(lambda msg: print(msg))
        self.worker.finished_signal.connect(self.show_results)
        self.worker.start()

    def show_results(self, result_img):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Start")
        
        if result_img is not None:
            cv2.imshow("Wynik detekcji drzew", result_img)
            cv2.waitKey(0)
        else:
            QMessageBox.information(self, "Koniec", "Analiza zakończona, ale nie wygenerowano obrazu z wynikami.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
