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
    url(r'raw', chatbot_main.view_api.API_raw),
    url(r'parsed', chatbot_main.view_api.API_parsed),
    url(r'pos', chatbot_main.view_api.API_pos),
    url(r'show', chatbot_main.view_basic.epy_view),
    url(r'test_sca', chatbot_main.view_basic.test_sca_view),
    url(r'chat', chatbot_main.view_basic.chat_view),
    url(r'^$', chatbot_main.view_basic.basic_view),
]


# from django.contrib import admin
# from django.urls import path
# from django.conf.urls import url, include
# import ui
# urlpatterns = [
#     # path('django/', admin.site.urls),
#     url(r'login', ui.views.login),
#     url(r'logout', ui.views.logout),
#     url(r'register', ui.views.register),
#     url(r'xdmin', ui.xdmin.Xdmin),
#     url(r'admin', ui.views.adminPage),
#     url(r'^$', ui.views.general),
# ]