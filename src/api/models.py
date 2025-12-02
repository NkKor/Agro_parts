# src/api/models.py
"""Модели данных для API"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from enum import Enum

class PredictRequest(BaseModel):
    """Модель запроса для /predict.
    Используется для документации, так как данные приходят как multipart/form-data.
    """
    pass

class PredictResponse(BaseModel):
    """Модель ответа для /predict"""
    pred_classes: List[str] = Field(..., example=["PART_001", "PART_002"])
    rank: Dict[str, int] = Field(..., example={"PART_001": 1, "PART_002": 2})
    more_possible_classes: List[str] = Field(..., example=["PART_001"])
    another_possible_classes: List[str] = Field(..., example=["PART_002"])
    similarities: Dict[str, float] = Field(..., example={"PART_001": 98.5, "PART_002": 92.1})

class CheckResponse(BaseModel):
    """Модель ответа для /check"""
    message: Dict[str, str] = Field(..., example={"PART_001": "Class found"})

class GetInfoResponse(BaseModel):
    """Модель ответа для /get_info"""
    message: Union[Dict[str, Any], List[str]] = Field(..., example={"available_classes": ["PART_001", "PART_002"]})

# Модель для параметров запроса
class PredictQueryParams(BaseModel):
    """Параметры запроса для /predict"""
    top_k: int = Field(default=5, ge=1, le=1000, description="Количество результатов для возврата (1-1000). По умолчанию 5.")