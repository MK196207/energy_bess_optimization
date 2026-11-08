from src.utils.config import BESSConfig

def load_bess_config() -> BESSConfig:
    """
    Wczytuje i waliduje konfiguracje magazynu energii BESS.
    """
    return BESSConfig.from_json()

if __name__ == "__main__":
    config = load_bess_config()
    print("Konfiguracja zaladowana pomyslnie:", config)
