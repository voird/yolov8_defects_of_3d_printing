# Generated by Django 5.0.4 on 2024-05-06 13:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('YoloObjectDetectionAPI', '0004_delete_processedimage'),
    ]

    operations = [
        migrations.AddField(
            model_name='imageupload',
            name='processed_timestamp',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]