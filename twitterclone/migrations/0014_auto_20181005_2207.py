# Generated by Django 2.1.1 on 2018-10-05 22:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('twitterclone', '0013_auto_20181004_2108'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='following',
            field=models.TextField(default='vic', max_length=1000),
        ),
    ]
