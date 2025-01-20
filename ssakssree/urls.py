
from django.contrib import admin
from django.urls import path, include
from rest_framework.views import APIView
from rest_framework.response import Response



class HealthCheckView(APIView):
    def get(self, request):
        return Response({"status": "ok"})


urlpatterns = [
    path('admin/', admin.site.urls),
    path('stats/', include('stats.urls')),
    path('', HealthCheckView.as_view(), name='health-check'),
]
