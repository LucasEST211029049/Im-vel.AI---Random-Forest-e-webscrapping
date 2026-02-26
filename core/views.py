# core/views.py
from django.shortcuts import render
from .utils import ImovelPredictor


def index(request):
    resultado = None
    erro = None

    if request.method == 'POST':
        # Captura dados do formulário
        try:
            dados = {
                'operacao': request.POST.get('operacao'),
                'uf': request.POST.get('uf'),
                'cidade_busca': request.POST.get('cidade_busca'),  # Slug para URL (ex: aguas-claras)
                'tipo_imovel': request.POST.get('tipo_imovel'),
                'metragem': float(request.POST.get('metragem')),
                'quartos': int(request.POST.get('quartos')),
                'suites': int(request.POST.get('suites')),
                'vagas': int(request.POST.get('vagas')),
                'bairro_preferencia': request.POST.get('bairro_preferencia'),  # Para One-Hot
            }

            predictor = ImovelPredictor()
            resultado = predictor.predict(dados)

            if "error" in resultado:
                erro = resultado["error"]
                resultado = None

        except ValueError:
            erro = "Por favor, preencha todos os campos numéricos corretamente."
        except Exception as e:
            erro = f"Ocorreu um erro inesperado: {str(e)}"

    return render(request, 'core/index.html', {'resultado': resultado, 'erro': erro})