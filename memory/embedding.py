
from typing import Union , Optional
import threading


class EmbeddingModel :
    def encode(self , texts : Union[str , list[str]]):
        raise NotImplementedError
    
    @property
    def demension(self) -> int : 
        raise NotImplementedError
    

_embeder : Optional[EmbeddingModel] = None
_lock = threading.Rlock()


def _build_embedder() -> EmbeddingModel : 
    os.getenv()


def get_text_embedder() -> EmbeddingModel:

    global _embedder
    if _embeder is not None:
        return _embeder
    with _lock : 
        if _embeder is None :
            _embeder = 


def get_dimension(default : int = 384) -> int :
    try :
        return int()