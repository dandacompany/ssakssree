from django.urls import path

from stats.views import ConjointView

urlpatterns = [
    path('conjoint/', ConjointView.as_view(), name='conjoint'),
]

