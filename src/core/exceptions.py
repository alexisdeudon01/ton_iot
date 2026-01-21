"""Exceptions personnalisées pour le pipeline TON IoT"""

class PipelineException(Exception):
    """Exception de base pour tout le pipeline"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class DataLoadingError(PipelineException):
    """Erreur lors du chargement des données"""
    pass

class InsufficientMemoryError(PipelineException):
    """RAM insuffisante pour l'opération"""
    def __init__(self, required_mb: float, available_mb: float):
        super().__init__(
            f"RAM insuffisante: besoin {required_mb:.1f}MB, disponible {available_mb:.1f}MB",
            details={'required_mb': required_mb, 'available_mb': available_mb}
        )

class ModelTrainingError(PipelineException):
    """Erreur lors de l'entraînement d'un modèle"""
    def __init__(self, model_name: str, original_error: Exception):
        super().__init__(
            f"Échec entraînement {model_name}: {str(original_error)}",
            details={'model': model_name, 'original': str(original_error)}
        )

class ConfigurationError(PipelineException):
    """Erreur de configuration"""
    pass

class ValidationError(PipelineException):
    """Erreur lors de la validation ou du tuning"""
    pass
