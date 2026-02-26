# core/utils.py
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class ImovelPredictor:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()

    def extrair_numero(self, texto):
        texto = str(texto)
        num = re.findall(r'\d+', texto)
        return int(num[0]) if num else 0

    def dividir_endereco(self, texto):
        if not texto:
            return None, None, None
        partes = [p.strip() for p in texto.split(",")]
        if len(partes) >= 3:
            return partes[0], partes[1], partes[2]
        elif len(partes) == 2:
            return partes[0], partes[1], None
        return partes[0], None, None

    def gerar_url(self, operacao, uf, cidade, tipo, quartos=None):
        # Normalização simples para slugs (pode ser melhorada com biblioteca slugify)
        cidade_slug = cidade.lower().replace(" ", "-").replace("á", "a").replace("ã", "a").replace("ç", "c")
        tipo_slug = tipo.lower()

        url = f"https://www.dfimoveis.com.br/{operacao}/{uf}/{cidade_slug}/{tipo_slug}"
        if quartos:
            url += f"/{quartos}-quartos"

        return url

    def predict(self, dados_input):
        # 1. Construir URL dinâmica
        url_base = self.gerar_url(
            dados_input['operacao'],
            dados_input['uf'],
            dados_input['cidade_busca'],
            dados_input['tipo_imovel']
        )

        print(f"Scraping URL: {url_base}")

        try:
            res = self.scraper.get(url_base, timeout=10)
            if res.status_code != 200:
                return {"error": "Não foi possível acessar o DF Imóveis. Verifique os parâmetros."}

            soup = BeautifulSoup(res.text, "html.parser")

            # Lógica de Paginação
            h1 = soup.find("h1")
            qtde_apartamentos = 0
            paginas = 0

            if h1:
                texto = h1.text.strip()
                numero = re.findall(r'\d[\d\.]*', texto)
                if numero:
                    qtde_apartamentos = int(numero[0].replace('.', ''))
                    paginas = qtde_apartamentos // 30

            if paginas == 0:
                paginas = 1  # Tenta pelo menos a primeira página

            dados = []

            # Limite de páginas para não demorar muito na web (max 5 para demo)
            for i in range(1, min(paginas, 5) + 1):
                url = f"{url_base}?pagina={i}"
                res = self.scraper.get(url,timeout=10)
                soup = BeautifulSoup(res.text, "html.parser")

                cards = soup.find_all("div", class_="imovel-info")  # Classe simplificada para busca

                # Fallback se a classe mudar, busca divs genericas de card
                if not cards:
                    cards = soup.find_all("div", id="resultado-pesquisa")

                # Reutilizando sua lógica de extração
                # Nota: A classe CSS do site pode mudar, mantive a sua original
                class_card_original = "imovel-info d-flex flex-column justify-content-between p-1 p-md-2 gap-0 gap-md-2 w-100 overflow-hidden"
                cards = soup.find_all("div", class_=class_card_original)

                for card in cards:
                    try:
                        enderecocomp = card.find("h2", class_="ellipse-text")
                        endereco, bairro, cidade_nome = self.dividir_endereco(
                            enderecocomp.text.strip() if enderecocomp else None)

                        precos = card.find_all("span", class_="body-large bold")
                        valor_imovel = None
                        if len(precos) >= 1:
                            v = precos[0].text.strip().replace(".", "").replace(",", "").replace("R$", "").strip()
                            if v.isdigit(): valor_imovel = float(v)

                        metragem_div = card.find("div",
                                                 class_="border-1 py-0 px-2 bg-white body-small rounded-pill web-view")
                        metragem = self.extrair_numero(metragem_div.text) if metragem_div else 0

                        item = card.find_all("div", class_="border-1 py-0 px-2 bg-white body-small rounded-pill")
                        qtde_quartos = 0
                        qtde_suites = 0
                        qtde_vagas = 0

                        for info in item:
                            texto_info = info.text.strip()
                            if "Quarto" in texto_info:
                                qtde_quartos = self.extrair_numero(texto_info)
                            elif "Suíte" in texto_info:
                                qtde_suites = self.extrair_numero(texto_info)
                            elif "Vaga" in texto_info:
                                qtde_vagas = self.extrair_numero(texto_info)

                        if valor_imovel:
                            dados.append({
                                "Bairro": bairro if bairro else "Centro",
                                "Cidade": cidade_nome if cidade_nome else dados_input['cidade_busca'],
                                "Metragem (m²)": metragem,
                                "Quartos": qtde_quartos,
                                "Suítes": qtde_suites,
                                "Vagas": qtde_vagas,
                                "Valor Imóvel (R$)": valor_imovel,
                            })
                    except Exception as e:
                        continue

            if len(dados) < 5:
                return {"error": "Poucos imóveis encontrados para treinar o modelo. Tente uma busca mais ampla."}

            # --- Processamento de Dados ---
            df = pd.DataFrame(dados)

            # Tratamento de Outliers
            p_01 = df["Valor Imóvel (R$)"].quantile(0.05)
            p_99 = df["Valor Imóvel (R$)"].quantile(0.95)
            df = df[(df["Valor Imóvel (R$)"] >= p_01) & (df["Valor Imóvel (R$)"] <= p_99)]

            X = df.drop(columns=["Valor Imóvel (R$)"])
            y = df["Valor Imóvel (R$)"]

            # One Hot Encoding
            X = pd.get_dummies(X, columns=["Bairro", "Cidade"])

            # Treino
            modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            modelo.fit(X, y)

            # Previsão
            imovel_usuario = {
                "Bairro": dados_input.get('bairro_preferencia', 'Centro'),
                "Cidade": dados_input.get('cidade_real', dados_input['cidade_busca']),
                # Caso o scraping retorne nomes diferentes
                "Metragem (m²)": dados_input['metragem'],
                "Quartos": dados_input['quartos'],
                "Suítes": dados_input['suites'],
                "Vagas": dados_input['vagas']
            }

            df_novo = pd.DataFrame([imovel_usuario])
            df_novo = pd.get_dummies(df_novo, columns=["Bairro", "Cidade"])

            # Alinhamento de colunas (Crucial!)
            df_novo_aligned = df_novo.reindex(columns=X.columns, fill_value=0)

            preco_previsto = modelo.predict(df_novo_aligned)[0]

            return {
                "success": True,
                "preco": preco_previsto,
                "qtd_imoveis_analisados": len(df),
                "r2_score": modelo.score(X, y)  # Apenas indicativo em treino total
            }

        except Exception as e:
            return {"error": f"Erro interno: {str(e)}"}