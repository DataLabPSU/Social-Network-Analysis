# Generated by Django 2.1.2 on 2018-10-07 18:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('twitterclone', '0016_profile_credibilityscore'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='real',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='profile',
            name='fake',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='profile',
            name='real',
            field=models.IntegerField(default=0),
        ),
    ]
