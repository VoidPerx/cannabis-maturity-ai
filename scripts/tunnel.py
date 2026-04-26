"""
Cloudflare Tunnel — expone http://localhost:8080 al mundo.
No requiere cuenta ni token. URL pública válida 24h.

Uso:
    python scripts/tunnel.py

La URL que imprime va en: GitHub Secrets → BACKEND_API_URL
"""
import subprocess, sys, re, time

def main():
    try:
        result = subprocess.run(["cloudflared", "--version"], capture_output=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("cloudflared no está instalado.")
        print("Descárgalo en: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
        sys.exit(1)

    print("Iniciando túnel hacia http://localhost:8080 ...")
    print("Presiona Ctrl+C para detener.\n")

    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8080"],
        stderr=subprocess.PIPE, text=True
    )

    url_found = False
    for line in proc.stderr:
        print(line, end="")
        if not url_found:
            m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
            if m:
                url = m.group(0)
                url_found = True
                print(f"\n{'='*60}")
                print(f"  URL PUBLICA: {url}")
                print(f"  Copia esta URL en GitHub Secrets → BACKEND_API_URL")
                print(f"{'='*60}\n")

    proc.wait()

if __name__ == "__main__":
    main()
