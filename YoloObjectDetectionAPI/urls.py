from django.urls import path
from .views import UploadImageView, process_image, index, upload_image, upload_i

urlpatterns = [
    path('', index, name='index'),
    path('process_image/', upload_image, name='process_image'),
    path('upload/', UploadImageView.as_view(), name='upload_image'),
    path('upload_image/', upload_i, name='upload_image_i'),
]
