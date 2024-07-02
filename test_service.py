import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import io

client = TestClient(app)

''' ===================== /face ===================== '''

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

@pytest.fixture
def mock_face_detection_service():
    with patch('app.routers.service.FaceDetectionService') as MockService:
        mock = MockService.return_value
        # Simular detección de una cara
        mock.detect_faces.return_value = [{'box': [100, 100, 50, 50], 'confidence': 0.95, 'keypoints': {}}]
        yield mock

def test_face_endpoint_stress():
    #sizes = [(i, i) for i in range(50, 5000, 50)] # Generar una lista de tamaños desde 50x50 hasta 5000x5000 en incrementos de 50
    #sizes = [(i, i) for i in range(100, 5000, 100)] # Generar una lista de tamaños desde 100x100 hasta 5000x5000 en incrementos de 100
    sizes = [(i, i) for i in range(100, 5000, 1000)]
    
    for size in sizes:
        image = Image.new('RGB', size, (255, 255, 255))
        with io.BytesIO() as img_buffer:
            image.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            response = client.post("/v1/face", files={"file": ("test_image.jpg", img_buffer, "image/jpeg")})
            print(f"Size: {size}, Response: {response.text}")
            assert response.status_code in [200, 400], f"Failed at size {size} with status code {response.status_code}"

# Se verifica que la preparación de la imagen no permita archivos corruptos 
def test_face_endpoint_empty_or_corrupt_file():
    # Test empty file
    empty_file = io.BytesIO(b"")
    response = client.post("/v1/face", files={"file": ("empty.jpg", empty_file, "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["detail"] == "El archivo está vacío."

    # Test corrupt file (invalid image data)
    corrupt_file = io.BytesIO(b"corrupt content")
    response = client.post("/v1/face", files={"file": ("corrupt.jpg", corrupt_file, "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["detail"] == "No se pudo decodificar la imagen."

#Se valida bloqueo en imagenes con formatos alternor a los permitidos 
def test_unsupported_mime_type():
    with open('tests/test_image.jpg', 'rb') as file:
        response = client.post("/v1/face", files={"file": ("test_image.tiff", file, "image/tiff")})
    assert response.status_code == 415
    assert response.json() == {'detail': 'Solo se aceptan archivos de imagen en formato JPEG, JPG o PNG'}



''' ===================== /embeddings ===================== '''

# Test para validar el endpoint de embeddings con una imagen válida
def test_embeddings_endpoint_valid_image():
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
            response = client.post("/v1/embeddings", files=files)
        
            # Verificar el código de estado y los valores retornados
            assert response.status_code == 200, f"Falló la solicitud para {image_file}. Código de estado: {response.status_code}"
            assert len(response.json()) > 0


# Aqui estoy validando que el servidor tenga suficientes recursos (memoria y CPU) para manejar la carga de imágenes grandes y numerosas solicitudes
# Genera una lista de tamaños que va desde 100x100 hasta 5000x5000 en incrementos de 100 píxeles. Esto crea un total de 50 tamaños diferentes.
# Al mismo tiempo se valida que el componenre retorne un 200 o 400 pero que no entre en excepción
def test_face_embeddings_endpoint_stress():
    #sizes = [(i, i) for i in range(50, 5000, 50)] # Generar una lista de tamaños desde 50x50 hasta 5000x5000 en incrementos de 50
    #sizes = [(i, i) for i in range(100, 5000, 100)] # Generar una lista de tamaños desde 100x100 hasta 5000x5000 en incrementos de 100
    sizes = [(i, i) for i in range(100, 5000, 1000)]
    
    for size in sizes:
        image = Image.new('RGB', size, (255, 255, 255))
        with io.BytesIO() as img_buffer:
            image.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            response = client.post("/v1/embeddings", files={"file": ("test_image.jpg", img_buffer, "image/jpeg")})
            print(f"Size: {size}, Response: {response.text}")
            assert response.status_code in [200, 400], f"Failed at size {size} with status code {response.status_code}"

#No necesitamos imagenes sin contenido
def test_empty_file():
    empty_file = io.BytesIO(b"")
    response = client.post("/v1/face", files={"file": ("empty.jpg", empty_file, "image/jpeg")})
    
    assert response.status_code == 400
    assert response.json()["detail"] == "El archivo está vacío."

# Validar que la imagen se pudo decodificar correctamente y no permitir scripts
def test_malicious_file():
    malicious_content = b"malicious content or script"
    file = io.BytesIO(malicious_content)
    response = client.post("/v1/face", files={"file": ("malicious.jpg", file, "image/jpeg")})
    
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "No se pudo decodificar la imagen." in response.json()["detail"]



def test_no_faces_detected():
    directory = 'tests/imgs_without_faces/'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as file:
                response = client.post("/v1/face", files={"file": (filename, file, "image/jpeg")})

            print(f"Testing {filename}: {response.status_code}, Response: {response.json()}")
            assert response.status_code == 400
            
            assert response.json()["detail"] in [
                'No se detectó ninguna cara en la imagen.',
                'La imagen no contiene caras humanas válidas.'
            ]



#Se valida bloqueo en imagenes con tamaños superiores a los permitidos
# def test_image_exceeding_max_file_size():
#     # Ajustar el tamaño de la imagen para que supere el límite de 15MB usando formato PNG
#     large_image = Image.new('RGB', (42000, 42000), (255, 255, 255))
#     with io.BytesIO() as img_buffer:
#         large_image.save(img_buffer, format='PNG')
        
#         # Verificar el tamaño del archivo después de guardarlo
#         img_buffer_size = img_buffer.tell()
#         assert img_buffer_size > 10 * 1024 * 1024, f"El archivo no es lo suficientemente grande: {img_buffer_size} bytes"
#         print(f"Tamaño del archivo: {img_buffer_size} bytes")
        
#         # Volver al inicio del buffer para leer su contenido
#         img_buffer.seek(0)
#         response = client.post("/v1/face", files={"file": ("large_image.png", img_buffer, "image/png")})
    
#     assert response.status_code == 413
#     assert response.json() == {'detail': 'El tamaño del archivo excede el límite permitido.'}
