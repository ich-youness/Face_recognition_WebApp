from django.urls import path
from .views import index,  predict_face

urlpatterns =[
    path("predict_face/" , predict_face, name="predict_face"),
    path("", index , name="index" )
]

