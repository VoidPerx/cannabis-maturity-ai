"""
Configuración inicial del proyecto.
Ejecutar una sola vez: python scripts/setup.py
"""
import shutil
from pathlib import Path

def main():
    root = Path(__file__).parent.parent

    # 1. Copiar .env si no existe
    env_example = root / ".env.example"
    env_file = root / ".env"
    if not env_file.exists():
        shutil.copy(env_example, env_file)
        print("[OK] .env creado - edita ROBOFLOW_API_KEY antes de continuar")
    else:
        print("[OK] .env ya existe")

    # 2. Crear directorios de datos
    for d in ["data/raw", "data/processed", "data/models", "data/exports", "data/mlruns"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    print("[OK] Directorios data/ creados")

    # 3. Inicializar DB
    import sys
    sys.path.insert(0, str(root))
    from app.db.database import init_db
    init_db()
    print("[OK] Base de datos SQLite inicializada en data/cannabis_maturity.db")

    print("\n[LISTO] Setup completo. Para iniciar:")
    print("   uvicorn app.main:app --reload --port 8000")
    print("   Abre http://localhost:8000\n")

if __name__ == "__main__":
    main()
