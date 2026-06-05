# Backend app package

# Monkeypatch for torchaudio to support DeepFilterNet on newer torchaudio versions
try:
    import sys
    import types
    import torchaudio
    
    # Check if the deprecated backend/common modules are missing
    if not hasattr(torchaudio, "backend") or not hasattr(torchaudio.backend, "common"):
        # Create a mock module for torchaudio.backend.common
        m = types.ModuleType("torchaudio.backend.common")
        # Map AudioMetaData to the new location at the root of torchaudio
        m.AudioMetaData = getattr(torchaudio, "AudioMetaData", None)
        
        # Inject the parent module if it doesn't exist
        if not hasattr(torchaudio, "backend"):
            backend_mod = types.ModuleType("torchaudio.backend")
            sys.modules["torchaudio.backend"] = backend_mod
            torchaudio.backend = backend_mod
            
        sys.modules["torchaudio.backend.common"] = m
        torchaudio.backend.common = m
except ImportError:
    pass
