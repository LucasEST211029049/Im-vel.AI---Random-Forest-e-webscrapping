from django.contrib import admin
from django.urls import path
from core import views  # Isso conecta com o nosso app

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'), # Isso diz que a rota vazia vai para a nossa tela
]