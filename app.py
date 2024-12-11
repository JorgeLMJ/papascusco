import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import mahotas as mh
from sklearn.cluster import KMeans
import onnxruntime as ort
from flask import Flask, render_template, request, send_from_directory, jsonify
from pathlib import Path
import uuid

app = Flask(__name__)

class ProcesadorPapaNativa:
    def __init__(self, modelo_onnx_path=None):
        self.caracteristicas = []
        self.etiquetas = []
        self.nombres_imagenes = []
        self.imagenes_fallidas = []
        self.modelo_onnx_path = modelo_onnx_path

        if modelo_onnx_path:
            # Cargar el modelo ONNX
            self.sesion = ort.InferenceSession(modelo_onnx_path)

    def preprocesar_imagen(self, imagen):
        max_dim = 800
        height, width = imagen.shape[:2]
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            imagen = cv2.resize(imagen, None, fx=scale, fy=scale)

        imagen = cv2.GaussianBlur(imagen, (5, 5), 0)
        return imagen

    def extraer_caracteristicas_forma(self, contorno):
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = float(w) / h
        extent = float(area) / (w * h) if (w * h) > 0 else 0
        circularidad = 4 * np.pi * area / (perimetro * perimetro) if perimetro > 0 else 0

        momentos = cv2.moments(contorno)
        hu_moments = cv2.HuMoments(momentos).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        return [area, perimetro, aspect_ratio, extent, circularidad] + list(hu_moments)

    def extraer_caracteristicas_color(self, imagen, mascara):
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)

        caracteristicas_color = []

        for espacio in [imagen, hsv, lab]:
            for i in range(3):
                canal = espacio[:, :, i]
                pixels = canal[mascara > 0]

                if len(pixels) > 0:
                    caracteristicas_color.extend([np.mean(pixels), np.std(pixels),
                                                 np.percentile(pixels, 25), np.percentile(pixels, 75)])
                else:
                    caracteristicas_color.extend([0, 0, 0, 0])

        return caracteristicas_color

    def extraer_caracteristicas_textura(self, imagen_gris, mascara):
        roi = imagen_gris.copy()
        roi[mascara == 0] = 0

        distancias = [1]
        angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm = graycomatrix(roi, distances=distancias, angles=angulos,
                            levels=256, symmetric=True, normed=True)

        propiedades = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

        caracteristicas_glcm = []
        for prop in propiedades:
            valor = graycoprops(glcm, prop).mean()
            caracteristicas_glcm.append(valor)

        haralick = mh.features.haralick(roi).mean(axis=0)[:13]

        return list(caracteristicas_glcm) + list(haralick)

    def extraer_color_predominante(self, imagen, n_colores=5):
        image = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        pixels = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=n_colores)
        kmeans.fit(pixels)

        colores = kmeans.cluster_centers_.astype(int)

        return colores.mean(axis=0)

    def contar_ojos(self, imagen, mascara):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        eye_count = 0
        for contorno in contornos:
            if cv2.contourArea(contorno) > 100:
                eye_count += 1

        return eye_count

    def predecir_con_modelo_onnx(self, caracteristicas):
        input_data = np.array([caracteristicas], dtype=np.float32)
        input_name = self.sesion.get_inputs()[0].name
        predicciones = self.sesion.run(None, {input_name: input_data})
        return predicciones[0]

    def procesar_imagen(self, ruta_imagen):
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            return None

        imagen = self.preprocesar_imagen(imagen)
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            return None

        contorno = max(contornos, key=cv2.contourArea)
        mascara = np.zeros(gris.shape, dtype=np.uint8)
        cv2.drawContours(mascara, [contorno], -1, 255, -1)

        caract_forma = self.extraer_caracteristicas_forma(contorno)
        caract_color = self.extraer_caracteristicas_color(imagen, mascara)
        caract_textura = self.extraer_caracteristicas_textura(gris, mascara)
        color_predominante = self.extraer_color_predominante(imagen)
        num_ojos = self.contar_ojos(imagen, mascara)

        caracteristicas = caract_forma + caract_color + caract_textura + list(color_predominante) + [num_ojos]

        if self.modelo_onnx_path:
            prediccion = self.predecir_con_modelo_onnx(caracteristicas)
            return prediccion[0]
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    if file:
        filename = f"{uuid.uuid4().hex}.jpg"
        file_path = os.path.join('static', filename)
        file.save(file_path)

        # Procesar imagen con el modelo
        modelo_path = "model/random_forest_model.onnx"
        procesador = ProcesadorPapaNativa(modelo_onnx_path=modelo_path)
        prediccion = procesador.procesar_imagen(file_path)

        if prediccion:
            return jsonify({"prediction": prediccion, "image_url": f"/static/{filename}"})
        else:
            return jsonify({"error": "No se pudo realizar la predicción."})

    return jsonify({"error": "No se subió ninguna imagen."})

if __name__ == '__main__':
    app.run(debug=True)
