# -*- coding: utf-8 -*-
"""
patch_x402_mainnet_cdp_payto_env.py

Migra similarity-search-api de Base Sepolia (testnet, wallet A hardcodeada,
facilitator x402.org) a Base mainnet real, con 3 cambios puntuales acordados
con David:

1. payTo: sigue siendo Wallet A (0x70E9F8057bB50e31B6ee06958bCbbe7DE9DAa98f,
   la misma constante ya usada por similarity-search-api/ws/useful-data-source
   -- Wallet A recibe, Wallet B solo paga/prueba, no se toca cual es cual) --
   pero deja de estar hardcodeada en el .py. Se lee de una env var nueva,
   NEXUS_X402_PAYTO_ADDRESS, decidido en sesion anterior para no exponer la
   direccion en el historial de git y poder rotarla sin redeploy. Falla al
   arrancar (os.environ[...], no .get()) si la env var no esta seteada --
   deliberado: mejor un crash-at-boot explicito que bootear silenciosamente
   sin payTo real en un asset que cobra dinero real.
2. Network: eip155:84532 (Base Sepolia) -> eip155:8453 (Base mainnet). Ya
   estaba anotado como pendiente en el comentario original de la linea (ver
   OLD_BLOCK) -- primera vez que se aplica.
3. Facilitator: x402.org/facilitator -> CDP Facilitator (mismo patron ya
   aplicado a "ws", archive/patches/patch_cdp_facilitator_bazaar_ws.py, en
   el repo nexus). create_facilitator_config() lee CDP_API_KEY_ID/
   CDP_API_KEY_SECRET del entorno -- deben estar seteadas en Railway ANTES
   de este deploy (plan acordado: reusar las de "ws" primero; si el primer
   verify/settle real falla, diagnosticar la causa real antes de asumir que
   hacen falta credenciales nuevas -- CDP+Base mainnet es una combinacion
   nueva en este repo, "ws" solo la valido en testnet).

A diferencia del patch de "ws", este NO agrega la extension Bazaar/discovery
(nadie la pidio para este asset en esta migracion) -- solo el extra minimo:
cdp-sdk en requirements.txt, sin tocar el pin de x402[evm,fastapi,mcp]==2.15.0
(misma version ya confirmada compatible con cdp.x402.create_facilitator_config
en patch_cdp_facilitator_bazaar_useful_data_source.py/patch_cdp_facilitator_bazaar_ws.py).

Uso:
    python patch_x402_mainnet_cdp_payto_env.py --file core/similarity_search_api_api.py --requirements requirements.txt

Patron establecido (CLAUDE.md S4): backup .bak, ast.parse antes/despues, match
exacto, idempotencia, grep de verificacion.
"""

import argparse
import ast
import shutil
import sys
from pathlib import Path

IDEMPOTENCY_MARKER = "NEXUS PATCH x402_mainnet_cdp_payto_env"

OLD_BLOCK = '''# --- NEXUS: x402 (pago por llamada en USDC, Base Sepolia testnet) ---
from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http.types import RouteConfig
from x402.mechanisms.evm.exact import ExactEvmServerScheme
from x402.schemas import Network
from x402.server import x402ResourceServer

_NEXUS_X402_EVM_ADDRESS = "0x70e9f8057bb50e31b6ee06958bcbbe7de9daa98f"
_NEXUS_X402_NETWORK: Network = "eip155:84532"  # Base Sepolia (testnet) -- cambiar a eip155:8453 + facilitator mainnet para produccion
_NEXUS_X402_PRICE = "$0.01"

_nexus_x402_facilitator = HTTPFacilitatorClient(
    FacilitatorConfig(url="https://x402.org/facilitator")
)
_nexus_x402_server = x402ResourceServer(_nexus_x402_facilitator)
_nexus_x402_server.register(_NEXUS_X402_NETWORK, ExactEvmServerScheme())'''

# pay_to sigue siendo Wallet A (payTo real, sin cambiar de wallet) -- solo deja
# de estar hardcodeada. os.environ[...] (no .get()) para fallar en el boot si
# falta la env var, no en el primer request real. "os" ya esta importado mas
# arriba en el archivo (linea 12), no hace falta reimportarlo.
NEW_BLOCK = '''# --- NEXUS: x402 (pago por llamada en USDC, Base mainnet) ---
from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http.types import RouteConfig
from x402.mechanisms.evm.exact import ExactEvmServerScheme
from x402.schemas import Network
from x402.server import x402ResourceServer
# --- NEXUS PATCH x402_mainnet_cdp_payto_env ---
from cdp.x402 import create_facilitator_config as _nexus_cdp_create_facilitator_config

# payTo (Wallet A) sale de env var, no hardcodeado -- no exponer la direccion
# en el historial de git, permite rotarla sin redeploy. Debe estar seteada en
# Railway antes de este deploy; sin ella el proceso no arranca (fail-fast).
_NEXUS_X402_EVM_ADDRESS = os.environ["NEXUS_X402_PAYTO_ADDRESS"]
_NEXUS_X402_NETWORK: Network = "eip155:8453"  # Base mainnet
_NEXUS_X402_PRICE = "$0.01"

# CDP Facilitator en vez de x402.org -- create_facilitator_config() lee
# CDP_API_KEY_ID/CDP_API_KEY_SECRET del entorno (deben estar seteadas en
# Railway antes de este deploy, mismo criterio que "ws").
_nexus_x402_facilitator = HTTPFacilitatorClient(
    _nexus_cdp_create_facilitator_config()
)
_nexus_x402_server = x402ResourceServer(_nexus_x402_facilitator)
_nexus_x402_server.register(_NEXUS_X402_NETWORK, ExactEvmServerScheme())'''


def patch_source(target: Path) -> None:
    original = target.read_text(encoding="utf-8")

    try:
        ast.parse(original)
    except SyntaxError as e:
        print(f"[ERROR] {target}: no parsea antes del patch: {e}", file=sys.stderr)
        sys.exit(1)

    if IDEMPOTENCY_MARKER in original:
        print(f"[SKIP] {target}: mainnet + CDP + payTo por env var ya esta integrado.")
        return

    count = original.count(OLD_BLOCK)
    if count != 1:
        print(
            f"[ERROR] {target}: bloque de configuracion x402 no matchea exacto "
            f"(encontradas {count}, se esperaba 1). Abortando sin escribir.",
            file=sys.stderr,
        )
        sys.exit(1)

    backup = target.with_suffix(target.suffix + ".bak_mainnet_cdp")
    shutil.copy2(target, backup)
    print(f"[OK] Backup: {backup}")

    patched = original.replace(OLD_BLOCK, NEW_BLOCK, 1)

    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"[ERROR] {target}: el patch rompio la sintaxis: {e}", file=sys.stderr)
        shutil.copy2(backup, target)
        sys.exit(1)

    target.write_text(patched, encoding="utf-8", newline="\n")
    print(f"[OK] Patch aplicado: {target}")


def patch_requirements(req_path: Path) -> None:
    if not req_path.exists():
        print(f"[ERROR] No existe {req_path}", file=sys.stderr)
        sys.exit(1)
    content = req_path.read_text(encoding="utf-8")
    if "cdp-sdk" in content:
        print(f"[SKIP] {req_path}: cdp-sdk ya esta en requirements.txt.")
        return

    if not content.endswith("\n"):
        content += "\n"
    content += "cdp-sdk\n"

    backup = req_path.with_suffix(req_path.suffix + ".bak_mainnet_cdp")
    shutil.copy2(req_path, backup)
    print(f"[OK] Backup: {backup}")

    req_path.write_text(content, encoding="utf-8", newline="\n")
    print(f"[OK] requirements.txt actualizado: {req_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Ruta al .py del asset")
    parser.add_argument("--requirements", required=True, help="Ruta al requirements.txt del asset")
    args = parser.parse_args()

    target = Path(args.file)
    if not target.exists():
        print(f"[ERROR] No existe: {target}", file=sys.stderr)
        sys.exit(1)

    patch_source(target)
    patch_requirements(Path(args.requirements))

    final = target.read_text(encoding="utf-8")
    print("\n--- Verificacion ---")
    checks = [
        ('os.environ["NEXUS_X402_PAYTO_ADDRESS"]', "payTo leido de env var, no hardcodeado"),
        ('_NEXUS_X402_EVM_ADDRESS = "0x70e9f8057bb50e31b6ee06958bcbbe7de9daa98f"', False, "constante hardcodeada removida"),
        ('"eip155:8453"', "network apunta a Base mainnet"),
        ('"eip155:84532"', False, "network testnet removido"),
        ("from cdp.x402 import create_facilitator_config", "import de cdp-sdk"),
        ("_nexus_cdp_create_facilitator_config()", "facilitator usa CDP en vez de x402.org"),
        ('FacilitatorConfig(url="https://x402.org/facilitator")', False, "facilitator viejo removido"),
        (IDEMPOTENCY_MARKER, "marcador de idempotencia presente"),
    ]
    all_ok = True
    for check in checks:
        if len(check) == 3:
            needle, should_exist, label = check
        else:
            needle, label = check
            should_exist = True
        present = needle in final
        ok = present if should_exist else not present
        print(f"  [{'OK' if ok else 'FALTA'}] {label}")
        all_ok = all_ok and ok

    if not all_ok:
        print("\n[WARNING] Alguna verificacion fallo.", file=sys.stderr)
        sys.exit(1)
    print(
        "\n[DONE] Revisar diff. Antes de pushear: confirmar en Railway que "
        "NEXUS_X402_PAYTO_ADDRESS (Wallet A) y CDP_API_KEY_ID/CDP_API_KEY_SECRET "
        "(reusadas de 'ws') estan seteadas en el servicio similarity-search-api -- "
        "sin la primera el proceso no bootea, sin las segundas el primer "
        "verify/settle real falla con 401."
    )


if __name__ == "__main__":
    main()
