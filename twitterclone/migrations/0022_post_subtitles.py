# Generated by Django 2.1.1 on 2018-10-18 18:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('twitterclone', '0021_messages'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='subtitles',
            field=models.TextField(blank=True, null=True),
        ),
    ]
