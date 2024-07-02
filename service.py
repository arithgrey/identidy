import os
import numpy as np
import cv2 as cv
from fastapi import APIRouter, File, status, UploadFile
from fastapi.exceptions import HTTPException
from keras_facenet import FaceNet
from mtcnn import MTCNN
from pydantic import BaseModel
from typing import List

# Tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

router = APIRouter()

class Keypoints(BaseModel):
    left_eye: List[int]
    right_eye: List[int]
    nose: List[int]
    mouth_left: List[int]
    mouth_right: List[int]

class FaceDto(BaseModel):
    box: List[int]
    confidence: float
    keypoints: Keypoints

class FaceDetectionService:
    def __init__(self):
        self.detector = MTCNN()
    
    def detect_faces(self, img: np.ndarray):
        faces = self.detector.detect_faces(img)
        return faces

class FaceEmbeddingService:
    def __init__(self):
        self.embedder = FaceNet()
    
    def get_embeddings(self, face: np.ndarray):
        resized_face = cv.resize(face, (160, 160))
        face_array = resized_face.astype("float32")
        face_array = np.expand_dims(face_array, axis=0)
        embeddings = self.embedder.embeddings(face_array)
        return embeddings

class ImageService:
    @staticmethod
    def prepare_image(data: bytes):
        try:
            # Validar tamaño máximo del archivo (15 MB)
            if len(data) > 15 * 1024 * 1024:  # 15 MB en bytes
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
                    detail="El tamaño del archivo excede el límite permitido."
                    )
            
            # Validar archivo vacío
            if not data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail="El archivo está vacío."
                    )
            
            npimg = np.frombuffer(data, np.uint8)
            img = cv.imdecode(npimg, cv.IMREAD_COLOR)

            # Validar que la imagen se pudo decodificar correctamente y no permitir scrits
            if img is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No se pudo decodificar la imagen.")

            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            return img
        
        except HTTPException:
            raise  # Re-raise HTTPException to propagate it up
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error al procesar la imagen.")

def validate_content_type(content_type: str):
    allowed_types = ("image/jpeg", "image/jpg", "image/png")
    if content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Solo se aceptan archivos de imagen en formato JPEG, JPG o PNG",
        )

def validate_single_face(faces):
    if len(faces) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No se detectó ninguna cara en la imagen.")
    elif len(faces) > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La imagen debe contener solo una cara.")

def validate_confidence(confidence: float, threshold: float = 0.9):
    if confidence < threshold:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La confidence del modelo es insuficiente para continuar")

face_detection_service = FaceDetectionService()
face_embedding_service = FaceEmbeddingService()

@router.post("/v1/face", tags=["Face Service"])
async def face(file: UploadFile = File(...)) -> FaceDto:
    validate_content_type(file.content_type)
    img = ImageService.prepare_image(await file.read())
    
    faces = face_detection_service.detect_faces(img)
    validate_single_face(faces)
    face = faces[0]
    validate_confidence(face["confidence"])
    
    return FaceDto(**face)

@router.post("/v1/embeddings", tags=["Face Service"])
async def embeddings(file: UploadFile = File(...)) -> List[float]:
    validate_content_type(file.content_type)
    img = ImageService.prepare_image(await file.read())
    
    faces = face_detection_service.detect_faces(img)
    validate_single_face(faces)
    face = faces[0]
    validate_confidence(face["confidence"])
    
    x, y, w, h = face["box"]
    face_img = img[y:y+h, x:x+w]
    embeddings = face_embedding_service.get_embeddings(face_img)
    
    if embeddings.size == 0:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No se generaron embeddings de la imagen.")
    
    return embeddings[0].tolist()
