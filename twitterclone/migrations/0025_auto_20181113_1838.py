# Generated by Django 2.1.3 on 2018-11-13 18:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('twitterclone', '0024_auto_20181113_1825'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='credibilityscore',
            field=models.FloatField(default=0.1),
        ),
    ]
