# Generated by Django 2.1.3 on 2018-11-13 18:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('twitterclone', '0025_auto_20181113_1838'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='imagename',
            field=models.TextField(default='image0.jpeg'),
        ),
    ]
