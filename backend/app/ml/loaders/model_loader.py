class ModelLoader:
    def __init__(self, model_version: str = "heuristic-v1") -> None:
        self._model_version = model_version

    def load_model_version(self) -> str:
        return self._model_version
