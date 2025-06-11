import os
import shutil
import random

# Porcentagem para treino
PERCENT_TREINO = 0.7

# Caminhos principais
PASTA_BASE = r'C:\Users\osantana\data_science\projetos\tracking_drop\Fase_2\retrain_model\dataset\rotulagem\selecionadas2'
PASTA_IMAGENS = os.path.join(PASTA_BASE, "images")
PASTA_LABELS = os.path.join(PASTA_BASE, "labels")
PASTA_TRAIN = os.path.join(PASTA_BASE, "train_test_split", "train")
PASTA_TEST = os.path.join(PASTA_BASE, "train_test_split", "test")

def criar_pastas_estrutura():
    """Cria as pastas necessárias para train/test split."""
    for subpasta in [PASTA_TRAIN, PASTA_TEST]:
        os.makedirs(os.path.join(subpasta, "images"), exist_ok=True)
        os.makedirs(os.path.join(subpasta, "labels"), exist_ok=True)


def listar_arquivos_validos():
    """Lista imagens que possuem arquivo .txt correspondente."""
    return [
        f for f in os.listdir(PASTA_IMAGENS)
        if os.path.isfile(os.path.join(PASTA_IMAGENS, f)) and
        os.path.exists(os.path.join(PASTA_LABELS, f"{os.path.splitext(f)[0]}.txt"))
    ]


def copiar_arquivos(arquivos, destino_imgs, destino_lbls):
    """Copia imagens e labels para as pastas de destino."""
    for arquivo in arquivos:
        nome_base, _ = os.path.splitext(arquivo)

        caminho_img = os.path.join(PASTA_IMAGENS, arquivo)
        caminho_txt = os.path.join(PASTA_LABELS, f"{nome_base}.txt")

        if os.path.exists(caminho_img):
            shutil.copy(caminho_img, os.path.join(destino_imgs, arquivo))
        if os.path.exists(caminho_txt):
            shutil.copy(caminho_txt, os.path.join(destino_lbls, f"{nome_base}.txt"))


def executar_split():
    """Executa o train/test split e copia os arquivos."""
    criar_pastas_estrutura()
    
    arquivos = listar_arquivos_validos()
    random.shuffle(arquivos)

    total = len(arquivos)
    qtd_treino = int(total * PERCENT_TREINO)

    arquivos_treino = arquivos[:qtd_treino]
    arquivos_teste = arquivos[qtd_treino:]

    copiar_arquivos(arquivos_treino,
                    os.path.join(PASTA_TRAIN, "images"),
                    os.path.join(PASTA_TRAIN, "labels"))

    copiar_arquivos(arquivos_teste,
                    os.path.join(PASTA_TEST, "images"),
                    os.path.join(PASTA_TEST, "labels"))

    print("✅ Split concluído com sucesso!")
    print(f"📂 {len(arquivos_treino)} arquivos para treino")
    print(f"📁 {len(arquivos_teste)} arquivos para teste")

# Executa o script
if __name__ == "__main__":
    executar_split()
