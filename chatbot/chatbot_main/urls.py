"""chatbot_main URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
import chatbot_main.view_basic
import chatbot_main.view_api

urlpatterns = [
    path(r'admin/', admin.site.urls),
    path(r'data/', include('host_data.views')),
    path(r'line/', include('host_line.views')),
    url(r'^webhook', chatbot_main.view_api.chatbot_callback),
    url(r'^chatbot/callback', chatbot_main.view_api.chatbot_callback),
    url(r'^chatbot/', chatbot_main.view_basic.basic_view),
]
