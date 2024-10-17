from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ExampleModelViewSet

router = DefaultRouter()
router.register(r'example', ExampleModelViewSet)

urlpatterns = [
    path('', include(router.urls)),
]