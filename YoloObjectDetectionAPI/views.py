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
    image_urls = []

    for image in images:
        image_path_in_bucket = f"media/images/{image.image_file.name}"
        image_url = default_storage.url(image_path_in_bucket)
        image_urls.append(image_url)

    context = {'images': images, 'image_urls': image_urls}

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
                obj = ImageUpload.objects.get(id=image_upload.id)
                file_obj = obj.image_file  
                image = Image.open(file_obj.open())
                image_np = np.array(image) 
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                detection_result = self.detector.run_detection(image_path=image_cv, confidence_threshold=image_upload.confidence_threshold)
                image_upload.status = ImageUpload.STATUS_COMPLETED
                image_with_labels = draw_labels(image_cv, detection_result)
                _, encoded_image = cv2.imencode('.jpg', image_with_labels)
                labeled_image = encoded_image.tobytes()
                image_name = f"processed_{obj.image_file.name.split('/')[-1]}"
                image_path_in_bucket = f"processed_images/{image_name}"
                default_storage.save(image_path_in_bucket, ContentFile(labeled_image))

                obj.image_file.name = image_path_in_bucket
                obj.processed_timestamp = timezone.now()
                obj.save()

                for detection in detection_result:
                    Detections.objects.create(
                        object_detection=obj,
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

def draw_labels(image, detections):
    for det in detections:
        x_min, y_min, x_max, y_max = int(det['x_min']), int(det['y_min']), int(det['x_max']), int(det['y_max'])
        label = det['label']
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return image

def process_image(request):
    
    images = ImageUpload.objects.all()
    image_urls = []

    detector = YOLOv8Detector('last.pt')

    for image in images:
        image_path_in_bucket = f"media/images/{image.image_file.name}"
        image_file = default_storage.open(image_path_in_bucket)

        pil_image = Image.open(image_file)
        image_np = np.array(pil_image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
        detection_result = detector.run_detection(image_path=image_cv)
        image_with_labels = draw_labels(image_cv, detection_result)
        _, encoded_image = cv2.imencode('.jpg', image_with_labels)
        labeled_image = encoded_image.tobytes()
        image_name = f"processed_{image.image_file.name.split('/')[-1]}"
        processed_image_path = f"processed_images/{image_name}"
        default_storage.save(processed_image_path, ContentFile(labeled_image))
        processed_image_url = default_storage.url(processed_image_path)
        image_urls.append(processed_image_url)

    context = {'image_urls': image_urls}
    return render(request, 'process_image.html', context)

def upload_i(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']

        image_upload = ImageUpload.objects.create(
            image_file=uploaded_image,
            confidence_threshold=0.5, 
            status=ImageUpload.STATUS_PROCESSING,
        )

        try:
            pil_image = Image.open(uploaded_image)
            image_np = np.array(pil_image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            detector = YOLOv8Detector('last.pt')
            detection_result = detector.run_detection(image_path=image_cv)

            image_with_labels = draw_labels(image_cv, detection_result)
            _, encoded_image = cv2.imencode('.jpg', image_with_labels)
            labeled_image = encoded_image.tobytes()
            processed_image_name = f"processed_{uploaded_image.name}"
            processed_image_path = f"processed_images/{processed_image_name}"
            default_storage.save(processed_image_path, ContentFile(labeled_image))
            image_upload.processed_timestamp = timezone.now()
            image_upload.status = ImageUpload.STATUS_COMPLETED
            image_upload.save()
            for det in detection_result:
                Detections.objects.create(
                    object_detection=image_upload,
                    label=det['label'],
                    confidence=det['confidence'],
                    x_min=det['x_min'],
                    x_max=det['x_max'],
                    y_min=det['y_min'],
                    y_max=det['y_max'],
                )
            processed_image_url = default_storage.url(processed_image_path)

            return render(request, 'upload_image.html', {
                'image_url': processed_image_url,
                'detection_result': detection_result,
            })

        except Exception as e:
            image_upload.status = ImageUpload.STATUS_FAILED
            image_upload.save()
            return render(request, 'upload_image.html', {'error': str(e)})

    return render(request, 'upload_image.html')
