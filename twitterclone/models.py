from django.db import models
from django.utils import timezone
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
import uuid
# Create your models here.
from embed_video.fields import EmbedVideoField


class Message(models.Model):
	author = models.ForeignKey(User, on_delete=models.CASCADE)
	message = models.TextField()
	recipient = models.CharField(max_length=100)

	def __str(self):
		return self.message

class Post(models.Model):
	author = models.ForeignKey(User, on_delete=models.CASCADE)
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
	videoname = models.TextField(blank=True, null=True)
	video = EmbedVideoField(blank=True, null=True)
	videolabels = models.TextField(blank=True, null=False)
	subtitles = models.TextField(blank=True,null=True)
	title = models.CharField(max_length=100)
	text = models.TextField()
	created_date = models.DateTimeField(
		default=timezone.now)
	likes = models.IntegerField(default=0)
	real = models.IntegerField(default=1)
	sharecomment = models.TextField(default='')
	updated = models.DateTimeField(default=timezone.now)
	def publish(self):
		self.save()

	def __str__(self):
		return self.title

class Messages(models.Model):
	author = models.ForeignKey(User,on_delete=models.CASCADE)
	recipient = models.TextField(default='')
	text = models.TextField(default='')
	time = models.DateTimeField(default=timezone.now)

class Share(models.Model):
	shared = models.ForeignKey(User, blank=True, null=True, on_delete=models.CASCADE)
	postid = models.TextField()
	date = models.DateTimeField(default=timezone.now)
	comment = models.TextField(default='')

class Survey(models.Model):
	author = models.ForeignKey(User,on_delete=models.CASCADE)
	age = models.IntegerField()
	gender = models.TextField()
	suggestions = models.TextField()
	withdrawdata = models.BooleanField('I wish to have my data withdrawn', default=False)

class Comment(models.Model):
	post = models.ForeignKey(Post, blank=True, null=True, on_delete=models.CASCADE)
	text = models.TextField()
	author = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)

	created_date = models.DateTimeField(default=timezone.now)

class Profile(models.Model):
	user = models.OneToOneField(User, on_delete=models.CASCADE)
	bio = models.TextField(max_length=500, blank=True)
	following = models.TextField(max_length=1000, default='')
	liked = models.TextField(max_length=1000, blank=True)
	notifications = models.IntegerField(default=1)
	notificationsString = models.TextField(default='')
	credibilityscore = models.FloatField(default=0.1)
	labels = models.TextField(default='')
	fake = models.IntegerField(default=0)
	real = models.IntegerField(default=0)
	imagename = models.TextField(default='image0.jpeg')
	amazonid = models.UUIDField(primary_key=False, default=uuid.uuid4, editable=False)

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
	if created:
		Profile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
	instance.profile.save()
