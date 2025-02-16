"""wxcloudrun URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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

from wxcloudrun import views
from django.urls import path

urlpatterns = [
    path("api/user/register", views.register),
    path("api/user/login", views.login),
    path("api/detect/detect", views.detect),
    path("api/detect/detect_by_video", views.detect_by_video),
    path("api/detect/comment", views.comment),
    path("api/detect/history", views.history),
    path("api/detect/get_all", views.get_all),
    path("api/detect/clear", views.clear),
]
