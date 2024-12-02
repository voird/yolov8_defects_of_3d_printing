from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import ImageUpload, Detections
from .serializers import ImageUploadSerializer
from .detection_models.yolov8 import YOLOv8Detector 
import cv2
from django.utils import timezone
from django.core.files.base import ContentFile

def upload_image(request):
    current_date = timezone.now().date()
    images = ImageUpload.objects.filter(upload_timestamp__date=current_date)
    
    context = {'images': images}
    return render(request, 'process_image.html', context)


class UploadImageView(APIView):
    detector = YOLOv8Detector('last.pt')

    def post(self, request, format=None):

        def draw_labels(image, detections):
            for det in detections:
                x_min, y_min, x_max, y_max = int(det['x_min']), int(det['y_min']), int(det['x_max']), int(det['y_max'])
                label = det['label']
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            return image

        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image_upload = serializer.save()

            try:
                detection_result = self.detector.run_detection(image_path=image_upload.image_file.path, confidence_threshold=image_upload.confidence_threshold)
                image_upload.status = ImageUpload.STATUS_COMPLETED

                image_with_detections = cv2.imread(image_upload.image_file.path)
                image_with_labels = draw_labels(image_with_detections, detection_result)

                labeled_image = cv2.imencode('.jpg', image_with_labels)[1].tostring()
                image_upload.image_file.save(image_upload.image_file.name, ContentFile(labeled_image))

                image_upload.processed_timestamp = timezone.now()
                image_upload.save()

                for detection in detection_result:
                    Detections.objects.create(
                        object_detection=image_upload,
                        label=detection['label'],
                        confidence=detection['confidence'],
                        x_min=detection['x_min'],
                        x_max=detection['x_max'],
                        y_min=detection['y_min'],
                        y_max=detection['y_max']
                    )

            except Exception as e:
                image_upload.status = ImageUpload.STATUS_FAILED
                image_upload.save()
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({'detections': detection_result}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


    
def index(request):
    return render(request, 'index.html')

def process_image(request):
    return render(request, 'process_image.html')