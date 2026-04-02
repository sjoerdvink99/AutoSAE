class AutoSAEError(Exception):
    pass


class UnsupportedArchitectureError(AutoSAEError):
    pass


class ConceptCardNotFoundError(AutoSAEError):
    pass


class ModelNotLoadedError(AutoSAEError):
    pass


class ConceptNotLoadedError(AutoSAEError):
    pass


class IncompatibleCardError(AutoSAEError):
    pass
