from django.db import models
from django.utils import timezone
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
import uuid
# Create your models here.
class Message(models.Model):
	author = models.ForeignKey(User, on_delete = models.CASCADE)
	message = models.TextField()
	recipient = models.CharField(max_length=100)
	def __str(self):
		return self.message
class Post(models.Model):
	author = models.ForeignKey(User, on_delete=models.CASCADE)
	id = models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
	title = models.CharField(max_length=100)
	text = models.TextField()
	created_date = models.DateTimeField(
			default=timezone.now)
	likes = models.IntegerField(default=0)

	def publish(self):
		self.save()
	def __str__(self):
		return self.title
class Share(models.Model):
	shared = models.ForeignKey(User,blank=True, null=True, on_delete=models.CASCADE)
	postid = models.TextField()
	date = models.DateTimeField(default=timezone.now)
class Comment(models.Model):
	post = models.ForeignKey(Post,blank=True, null=True, on_delete=models.CASCADE)
	text = models.TextField()
	author = models.ForeignKey(User, on_delete=models.CASCADE,blank=True, null=True)

	created_date = models.DateTimeField(default=timezone.now)
class Profile(models.Model):
	user = models.OneToOneField(User, on_delete=models.CASCADE)
	bio = models.TextField(max_length=500,blank=True)
	following = models.TextField(max_length=1000,blank=True)
	liked = models.TextField(max_length=1000,blank=True)
	notifications = models.IntegerField(default=1)
@receiver(post_save,sender=User)
def create_user_profile(sender, instance, created, **kwargs):
	if created:
		Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender,instance,**kwargs):
	instance.profile.save()
