import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import io

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def create_test_images():
    # Crear una imagen aleatoria de 100x100 píxeles
    image_array = np.random.rand(100, 100, 3) * 255
    image_random = Image.fromarray(image_array.astype('uint8')).convert('RGB')

    # Crear una imagen de 100x100 píxeles con una cara simulada
    image_with_face = Image.new('RGB', (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(image_with_face)
    draw.rectangle((25, 25, 75, 75), fill=(255, 255, 255), outline=(0, 0, 0))  # Simular una cara rectangular
    
    # Guardar las imágenes en la carpeta 'tests'
    if not os.path.exists('tests'):
        os.makedirs('tests')
    image_random.save('tests/test_image.jpg')

    # Crear una imagen con una cara más realista
    image_realistic_face = Image.new('RGB', (100, 100), (255, 255, 255))
    draw_face = ImageDraw.Draw(image_realistic_face)
    draw_face.rectangle((25, 25, 75, 75), fill=(255, 255, 255), outline=(0, 0, 0))  # Simular una cara rectangular con baja confianza
    
    # Añadir características faciales más distintivas (ojos, nariz, boca)
    draw_face.rectangle((35, 30, 45, 40), fill=(0, 0, 0))  # Ojo izquierdo
    draw_face.rectangle((55, 30, 65, 40), fill=(0, 0, 0))  # Ojo derecho
    draw_face.rectangle((45, 50, 55, 60), fill=(0, 0, 0))  # Nariz
    draw_face.rectangle((40, 65, 60, 70), fill=(0, 0, 0))  # Boca
    
    image_realistic_face.save('tests/test_image_with_face.jpg')
    
    yield

    os.remove('tests/test_image.jpg')
    os.remove('tests/test_image_with_face.jpg')


@pytest.fixture
def mock_face_detection_service():
    with patch('app.routers.service.FaceDetectionService') as MockService:
        mock = MockService.return_value
        # Simular detección de una cara
        mock.detect_faces.return_value = [{'box': [100, 100, 50, 50], 'confidence': 0.95, 'keypoints': {}}]
        yield mock

@pytest.fixture
def mock_face_embedding_service():
    with patch('app.routers.service.FaceEmbeddingService') as MockService:
        mock = MockService.return_value
        # Simular obtención de embeddings
        mock.get_embeddings.return_value = [[0.1, 0.2, 0.3, 0.4]]
        yield mock
    
# El app debe regresar status 415 en el caso de que el formato de imagen sea uncorrecto
def test_invalid_image_format(mock_face_detection_service, mock_face_embedding_service):
    
    text_content = b"Este es un archivo de texto en lugar de una imagen"
    file = io.BytesIO(text_content)

    response = client.post("/v1/face", files={"file": ("filename.txt", file, "text/plain")})

    assert response.status_code == 415
    assert response.json() == {'detail': 'Solo se aceptan archivos de imagen en formato JPEG, JPG o PNG'}

    # Verificar que no se invocaron los servicios subyacentes
    assert not mock_face_detection_service.detect_faces.called
    assert not mock_face_embedding_service.get_embeddings.called


# Test para validar que lance el 200 al llegar al confidence
def test_face_endpoint_confidence():
    with open('tests/cara_mujer.jpg', 'rb') as file:
        response = client.post("/v1/face", files={"file": ("filename", file, "image/jpeg")})
    assert response.status_code == 200


# Test para validar que alcance un confidence mayor a .89
def test_face_endpoint_confidence_bigger():
    with open('tests/cara_mujer.jpg', 'rb') as file:
        response = client.post("/v1/face", files={"file": ("filename", file, "image/jpeg")})
    
    assert response.json()["confidence"] > 0.89


def test_face_endpoint_for_images():
    image_dir = "tests/imgs_frente"
    assert os.path.isdir(image_dir), f"El directorio {image_dir} no existe."

    # Obtener la lista de archivos de imagen en el directorio
    image_files = os.listdir(image_dir)
    assert image_files, f"No se encontraron imágenes en {image_dir}."

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)

        # Abrir la imagen y preparar para enviarla al endpoint
        with open(image_path, 'rb') as image_file:
            files = {'file': (image_file.name, image_file, 'image/jpeg')}
            
            # Hacer la solicitud POST al endpoint usando el cliente de prueba
            response = client.post("/v1/face", files=files)

            # Verificar el código de estado y los valores retornados
            assert response.status_code == 200, f"Falló la solicitud para {image_file}. Código de estado: {response.status_code}"
            
            # Verificar la estructura del JSON retornado
            json_response = response.json()
            assert 'box' in json_response, "No se encontró 'box' en la respuesta JSON."
            assert 'confidence' in json_response, "No se encontró 'confidence' en la respuesta JSON."
            assert 'keypoints' in json_response, "No se encontró 'keypoints' en la respuesta JSON."

# Crea un archivo de imagen simulado (Sin caras, debe regresar el 400 indicando que esta no es una cara)
def test_face_endpoint_with_no_faces(mock_face_detection_service):
    with open('tests/test_image.jpg', 'rb') as file:
        response = client.post("/v1/face", files={"file": ("filename", file, "image/jpeg")})

    assert response.status_code == 400
    assert response.json() == {'detail': 'No se detectó ninguna cara en la imagen.'}

#Se valida que sea capaz de identificar más de unas cara
def test_face_endpoint_with_many_faces(mock_face_detection_service):
    directory = 'tests/imgs_many_faces'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as file:
                response = client.post("/v1/face", files={"file": (filename, file, "image/jpeg")})

            print(f"Testing {filename}: {response.status_code}, Response: {response.json()}")
            assert response.status_code == 400
            
            assert response.json()["detail"] in [
                'La imagen debe contener solo una cara.',
                'No se detectó ninguna cara en la imagen.'
            ]
