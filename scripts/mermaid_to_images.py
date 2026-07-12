"""
Converte blocos ```mermaid em imagens PNG via Mermaid.ink API.
Gera um novo arquivo .md com as imagens no lugar dos blocos,
pronto para exportar como PDF sem problemas.

Uso:
    python scripts/mermaid_to_images.py <arquivo.md>

Exemplo:
    python scripts/mermaid_to_images.py documentos/arquitetura/README.md
"""

import re
import base64
import sys
import urllib.request
import urllib.error
from pathlib import Path


def encode_mermaid(code: str) -> str:
    """Codifica o código Mermaid em base64 para uso na URL do mermaid.ink."""
    encoded = base64.urlsafe_b64encode(code.encode("utf-8")).decode("utf-8")
    return encoded


def download_image(url: str, dest_path: Path) -> bool:
    """Baixa a imagem PNG do mermaid.ink e salva localmente."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            dest_path.write_bytes(resp.read())
        return True
    except urllib.error.URLError as e:
        print(f"  ⚠️  Erro ao baixar imagem: {e}")
        return False


def convert_md_mermaid(input_path: Path) -> Path:
    """
    Lê o markdown, converte blocos mermaid em imagens PNG baixadas localmente,
    e salva um novo arquivo _rendered.md no mesmo diretório.
    """
    content = input_path.read_text(encoding="utf-8")

    # Diretório para salvar as imagens geradas
    images_dir = input_path.parent / "mermaid_images"
    images_dir.mkdir(exist_ok=True)

    pattern = re.compile(r"```mermaid\n([\s\S]*?)```")
    counter = [0]

    def replace_block(match: re.Match) -> str:
        counter[0] += 1
        diagram_code = match.group(1).strip()
        encoded = encode_mermaid(diagram_code)
        url = f"https://mermaid.ink/img/{encoded}?type=png"
        img_filename = f"diagram_{counter[0]:02d}.png"
        img_path = images_dir / img_filename

        print(f"  → Diagrama {counter[0]}: baixando de mermaid.ink...")
        success = download_image(url, img_path)

        if success:
            # Caminho relativo para o markdown
            rel_path = f"mermaid_images/{img_filename}"
            print(f"     ✅ Salvo em: {rel_path}")
            return f"![Diagrama {counter[0]}]({rel_path})"
        else:
            # Fallback: usa URL direta (requer internet na hora do PDF)
            print("     ⚠️  Usando URL direta como fallback.")
            return f"![Diagrama {counter[0]}]({url})"

    new_content = pattern.sub(replace_block, content)

    output_path = input_path.parent / (input_path.stem + "_rendered.md")
    output_path.write_text(new_content, encoding="utf-8")

    print(f"\n✅ Arquivo gerado: {output_path}")
    print("   Agora abra este arquivo no VS Code e exporte como PDF.")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/mermaid_to_images.py <arquivo.md>")
        print("Exemplo: python scripts/mermaid_to_images.py documentos/arquitetura/README.md")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"❌ Arquivo não encontrado: {input_file}")
        sys.exit(1)

    print(f"📄 Processando: {input_file}")
    convert_md_mermaid(input_file)
